
import numpy as np
import cv2
import math
import argparse
import os
os.add_dll_directory(r"C:\Users\Aman.DESKTOP-VEB4T1B\AppData\Local\Programs\Python\Python312\Lib\site-packages\opencv\build\x64\vc16\bin")
import random

import torch
import torch.optim as optim
import ngransac

from network import CNNet
from dataset import SparseDataset
import util

parser = util.create_parser('NG-RANSAC demo for a user defined image pair. Fits an essential matrix (default) or fundamental matrix (-fmat) using OpenCV RANSAC vs. NG-RANSAC.')
parser.add_argument('--image1', '-img1', default='images/demo1.jpg', help='path to image 1')
parser.add_argument('--image2', '-img2', default='images/demo2.jpg', help='path to image 2')
parser.add_argument('--outimg', '-out', default='demo.png', help='demo will store a matching image under this file name')
parser.add_argument('--focallength1', '-fl1', type=float, default=900, help='focal length of image 1 (only used when fitting the essential matrix)')
parser.add_argument('--focallength2', '-fl2', type=float, default=900, help='focal length of image 2 (only used when fitting the essential matrix)')
parser.add_argument('--model', '-m', default='', help='model to load, leave empty and the script infers an appropriate pre-trained model from the other settings')
parser.add_argument('--hyps', '-hyps', type=int, default=1000, help='number of hypotheses, i.e. number of RANSAC iterations')
parser.add_argument('--refine', '-ref', action='store_true', help='refine using the 8point algorithm on all inliers, only used for fundamental matrix estimation (-fmat)')

opt = parser.parse_args()

if opt.fmat:
    print("\nFitting Fundamental Matrix...\n")
else:
    print("\nFitting Essential Matrix...\n")

# Detector setup (SIFT or ORB)
if getattr(opt, 'orb', False):
    print("Using ORB.\n")
    if getattr(opt, 'nfeatures', None) and opt.nfeatures > 0:
        detector = cv2.ORB_create(nfeatures=opt.nfeatures)
    else:
        detector = cv2.ORB_create()
else:
    if getattr(opt, 'rootsift', False):
        print("Using RootSIFT.\n")
    else:
        print("Using SIFT.\n")
    if getattr(opt, 'nfeatures', None) and opt.nfeatures > 0:
        detector = cv2.SIFT_create(nfeatures=opt.nfeatures, contrastThreshold=1e-5)
    else:
        detector = cv2.SIFT_create()

# Load neural guidance network
model_file = opt.model
if len(model_file) == 0:
    model_file = util.create_session_string('e2e', opt.fmat, getattr(opt, 'orb', False), getattr(opt, 'rootsift', False), getattr(opt, 'ratio', 1.0), getattr(opt, 'session', ''))
    model_file = 'models/weights_' + model_file + '.net'
    print("No model file specified. Inferring pre-trained model from given parameters:")
    print(model_file)

# Get resblocks from args (util.create_parser defaults to 12, but we want to allow override)
# If model file path contains 'synthetic', default to 2 resblocks (what we trained with)
if 'synthetic' in model_file.lower():
	resblocks = getattr(opt, 'resblocks', 2)
	print(f"Detected synthetic model, using {resblocks} resblocks")
else:
	resblocks = getattr(opt, 'resblocks', 12)  # Default for pre-trained models

model = CNNet(resblocks)
model.load_state_dict(torch.load(model_file))
model = model.cuda()
model.eval()
print("Successfully loaded model.")

print("\nProcessing pair:")
print("Image 1:", opt.image1)
print("Image 2:", opt.image2)

# Read images
img1 = cv2.imread(opt.image1)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread(opt.image2)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Camera calibration
K1 = np.eye(3)
K1[0,0] = K1[1,1] = opt.focallength1
K1[0,2] = img1.shape[1] * 0.5
K1[1,2] = img1.shape[0] * 0.5
K2 = np.eye(3)
K2[0,0] = K2[1,1] = opt.focallength2
K2[0,2] = img2.shape[1] * 0.5
K2[1,2] = img2.shape[0] * 0.5

# Detect features
kp1, desc1 = detector.detectAndCompute(img1, None)
kp2, desc2 = detector.detectAndCompute(img2, None)

print("\nFeature found in image 1:", len(kp1))
print("Feature found in image 2:", len(kp2))

# RootSIFT normalization if requested
if getattr(opt, 'rootsift', False):
    print("Using RootSIFT normalization.")
    desc1 = util.rootSift(desc1)
    desc2 = util.rootSift(desc2)

# Feature matching
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc1, desc2, k=2)

good_matches = []
pts1 = []
pts2 = []
ratios = []

print("")
if getattr(opt, 'ratio', 1.0) < 1.0:
    print("Using Lowe's ratio filter with", opt.ratio)

for (m, n) in matches:
    if m.distance < getattr(opt, 'ratio', 1.0) * n.distance: # Lowe's ratio filter
        good_matches.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
        ratios.append(m.distance / n.distance)

print("Number of valid matches:", len(good_matches))

# Robust dimensional handling
pts1 = np.asarray(pts1)        # shape (N, 2)
pts2 = np.asarray(pts2)        # shape (N, 2)
ratios = np.asarray(ratios)    # shape (N,)
if ratios.ndim == 1:
    ratios = ratios.reshape(-1, 1) # shape (N, 1)

# For essential matrix: undistort points, then reshape
if not opt.fmat:
    pts1 = cv2.undistortPoints(pts1, K1, None)
    pts2 = cv2.undistortPoints(pts2, K2, None)
    pts1 = pts1.reshape(-1, 2)
    pts2 = pts2.reshape(-1, 2)

# Build final correspondences array (N, 5)
correspondences = np.concatenate((pts1, pts2, ratios), axis=1) # shape (N, 5)

# ------------------------------------------------
# Fit fundamental or essential matrix using OpenCV
# ------------------------------------------------
if opt.fmat:
    ransac_model, ransac_inliers = cv2.findFundamentalMat(pts1, pts2, ransacReprojThreshold=getattr(opt, 'threshold', 0.001), confidence=0.999)
else:
    K = np.eye(3)
    ransac_model, ransac_inliers = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=getattr(opt, 'threshold', 0.001))

print("\n=== Model found by RANSAC: ==========\n")
print(ransac_model)
print("\nRANSAC Inliers:", ransac_inliers.sum())

# ---------------------------------------------------
# Fit fundamental or essential matrix using NG-RANSAC
# ---------------------------------------------------
if opt.fmat:
    util.normalize_pts(pts1, img1.shape)
    util.normalize_pts(pts2, img2.shape)

if getattr(opt, 'nosideinfo', False):
    ratios = np.zeros(ratios.shape)
    correspondences = np.concatenate((pts1, pts2, ratios), axis=1)

# Prepare tensor for neural model: (1, 5, N, 1)
corr_tensor = torch.from_numpy(correspondences).float()      # (N, 5)
model_input = corr_tensor.t().unsqueeze(0).unsqueeze(-1)     # (1, 5, N, 1)
log_probs = model(model_input.cuda())[0]
probs = torch.exp(log_probs).cpu()

out_model = torch.zeros((3, 3)).float()        # estimated model
out_inliers = torch.zeros(log_probs.size())    # inlier mask
out_gradients = torch.zeros(log_probs.size())  # gradient tensor (training only)
rand_seed = 0

# For NG-RANSAC extension: shape (1, N, 5)
corr_tensor_ngransac = corr_tensor.unsqueeze(0)              # (1, N, 5)

if opt.fmat:
    incount = ngransac.find_fundamental_mat(
        corr_tensor_ngransac.cpu(), probs, rand_seed, opt.hyps, getattr(opt, 'threshold', 0.001),
        opt.refine, out_model, out_inliers, out_gradients
    )
else:
    incount = ngransac.find_essential_mat(
        corr_tensor_ngransac.cpu(), probs, rand_seed, opt.hyps, getattr(opt, 'threshold', 0.001),
        out_model, out_inliers, out_gradients
    )

print("\n=== Model found by NG-RANSAC: =======\n")
print(out_model.numpy())
print("\nNG-RANSAC Inliers: ", int(incount))

# Visualization and saving
out_inliers = out_inliers.byte().numpy().ravel().tolist()
ransac_inliers = ransac_inliers.ravel().tolist()

match_img_ransac = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2, matchColor=(75,180,60), matchesMask=ransac_inliers)
match_img_ngransac = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2, matchColor=(200,130,0), matchesMask=out_inliers)
match_img = np.concatenate((match_img_ransac, match_img_ngransac), axis=0)

cv2.imwrite(opt.outimg, match_img)
print("\nDone. Visualization of the result stored as", opt.outimg)