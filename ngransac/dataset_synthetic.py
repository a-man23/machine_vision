import numpy as np
import torch
import os
import cv2
import math
import util

from torch.utils.data import Dataset

class SparseDatasetSynthetic(Dataset):
	"""Sparse correspondences dataset for synthetic .npz files."""

	def __init__(self, folders, ratiothreshold=1.0, nfeatures=-1, fmat=False, overwrite_side_info=False):

		self.nfeatures = nfeatures # ensure fixed number of features, -1 keeps original feature count
		self.ratiothreshold = ratiothreshold # threshold for Lowe's ratio filter
		self.overwrite_side_info = overwrite_side_info # if true, provide no side information to the neural guidance network
		
		# collect precalculated correspondences of all provided datasets
		self.files = []
		for folder in folders:
			if isinstance(folder, list):
				# If it's a file list, use it directly
				self.files += folder
			elif os.path.isdir(folder):
				self.files += [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith('.npz')]
			else:
				# If it's a single file path
				if os.path.isfile(folder) and folder.endswith('.npz'):
					self.files.append(folder)

		self.fmat = fmat # estimate fundamental matrix instead of essential matrix
		self.minset = 5 # minimal set size for essential matrices
		if fmat: self.minset = 7 # minimal set size for fundamental matrices
			
	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):

		# load precalculated correspondences from .npz file
		data = np.load(self.files[idx], allow_pickle=True)

		# Load data from .npz format: arr_0 to arr_8
		# arr_0: pts1 (1, N, 2)
		# arr_1: pts2 (1, N, 2)
		# arr_2: ratios (1, N, 1)
		# arr_3: im_size1 (2,)
		# arr_4: im_size2 (2,)
		# arr_5: K1 (3, 3)
		# arr_6: K2 (3, 3)
		# arr_7: R (3, 3)
		# arr_8: t (3, 1)
		
		pts1 = data['arr_0']  # (1, N, 2)
		pts2 = data['arr_1']  # (1, N, 2)
		ratios = data['arr_2']  # (1, N, 1)
		im_size1 = data['arr_3']  # (2,)
		im_size2 = data['arr_4']  # (2,)
		K1 = data['arr_5']  # (3, 3)
		K2 = data['arr_6']  # (3, 3)
		gt_R = data['arr_7']  # (3, 3)
		gt_t = data['arr_8']  # (3, 1)

		# Remove batch dimension if present
		if pts1.ndim == 3 and pts1.shape[0] == 1:
			pts1 = pts1[0]  # (N, 2)
		if pts2.ndim == 3 and pts2.shape[0] == 1:
			pts2 = pts2[0]  # (N, 2)
		if ratios.ndim == 3 and ratios.shape[0] == 1:
			ratios = ratios[0]  # (N, 1)

		# Ensure ratios is 2D (N, 1)
		if ratios.ndim == 1:
			ratios = ratios.reshape(-1, 1)

		# Reshape to match expected format: (1, N, 2) and (1, N, 1)
		pts1 = pts1.reshape(1, -1, 2)
		pts2 = pts2.reshape(1, -1, 2)
		ratios = ratios.reshape(1, -1, 1)

		# applying Lowes ratio criterion
		ratio_filter = ratios[0,:,0] < self.ratiothreshold

		if ratio_filter.sum() < self.minset: # ensure a minimum count of correspondences
			print("WARNING! Ratio filter too strict. Only %d correspondences would be left, so I skip it." % int(ratio_filter.sum()))
		else:
			pts1 = pts1[:,ratio_filter,:]
			pts2 = pts2[:,ratio_filter,:]
			ratios = ratios[:,ratio_filter,:]
		
		if self.overwrite_side_info:
			ratios = np.zeros(ratios.shape, dtype=np.float32)

		# Convert to torch tensors
		im_size1 = torch.from_numpy(np.asarray(im_size1)).float()
		im_size2 = torch.from_numpy(np.asarray(im_size2)).float()
		K1 = torch.from_numpy(K1).float()
		K2 = torch.from_numpy(K2).float()
		gt_R = torch.from_numpy(gt_R).float()
		gt_t = torch.from_numpy(gt_t).float()

		if self.fmat:
			# for fundamental matrices, normalize image coordinates using the image size (network should be independent to resolution)
			util.normalize_pts(pts1, im_size1)
			util.normalize_pts(pts2, im_size2)
		else:
			#for essential matrices, normalize image coordinate using the calibration parameters
			# cv2.undistortPoints expects (1, N, 2) and returns (N, 1, 2), so we need to reshape
			pts1_undist = cv2.undistortPoints(pts1, K1.numpy(), None)
			pts2_undist = cv2.undistortPoints(pts2, K2.numpy(), None)
			# Reshape back to (1, N, 2) format
			pts1 = pts1_undist.reshape(1, -1, 2)
			pts2 = pts2_undist.reshape(1, -1, 2)

		# Ensure all arrays have the same shape (1, N, ...) before concatenation
		# pts1: (1, N, 2), pts2: (1, N, 2), ratios: (1, N, 1)
		# stack image coordinates and side information into one tensor
		correspondences = np.concatenate((pts1, pts2, ratios), axis=2)
		correspondences = np.transpose(correspondences)
		correspondences = torch.from_numpy(correspondences).float()

		if self.nfeatures > 0:
			# ensure that there are exactly nfeatures entries in the data tensor 
			if correspondences.size(1) > self.nfeatures:
				rnd = torch.randperm(correspondences.size(1))
				correspondences = correspondences[:,rnd,:]
				correspondences = correspondences[:,0:self.nfeatures]

			if correspondences.size(1) < self.nfeatures:
				result = correspondences
				for i in range(0, math.ceil(self.nfeatures / correspondences.size(1) - 1)):
					rnd = torch.randperm(correspondences.size(1))
					result = torch.cat((result, correspondences[:,rnd,:]), dim=1)
				correspondences = result[:,0:self.nfeatures]

		# construct the ground truth essential matrix from the ground truth relative pose
		gt_E = torch.zeros((3,3))
		gt_E[0, 1] = -float(gt_t[2,0])
		gt_E[0, 2] = float(gt_t[1,0])
		gt_E[1, 0] = float(gt_t[2,0])
		gt_E[1, 2] = -float(gt_t[0,0])
		gt_E[2, 0] = -float(gt_t[1,0])
		gt_E[2, 1] = float(gt_t[0,0])

		gt_E = gt_E.mm(gt_R)

		# fundamental matrix from essential matrix
		gt_F = K2.inverse().transpose(0, 1).mm(gt_E).mm(K1.inverse())

		return correspondences, gt_F, gt_E, gt_R, gt_t, K1, K2, im_size1, im_size2

