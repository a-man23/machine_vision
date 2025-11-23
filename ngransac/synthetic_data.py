import numpy as np
import cv2
import os

def random_intrinsics(fx=800, fy=800, cx=320, cy=240):
    #Create a simple pinhole camera intrinsic matrix.
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    return K

def random_pose(max_angle_deg=30, max_translation=1.0):
    #Random rotation and translation.
    angle = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg, 3))
    Rx = cv2.Rodrigues(np.array([angle[0], 0, 0]))[0]
    Ry = cv2.Rodrigues(np.array([0, angle[1], 0]))[0]
    Rz = cv2.Rodrigues(np.array([0, 0, angle[2]]))[0]
    R = Rz @ Ry @ Rx

    t = np.random.uniform(-max_translation, max_translation, 3)
    t = t / np.linalg.norm(t)  # normalize direction
    t = t.reshape(3,1)

    return R.astype(np.float32), t.astype(np.float32)

def generate_3D_points(n):
    #Random points in front of the camera.
    pts = np.random.uniform(-1, 1, (n, 3))
    pts[:,2] += 3.0   # ensure positive depth
    return pts

def project(K, R, t, pts_3D):
    #Project 3D points into 2D image plane.
    P = K @ np.hstack((R, t))  # projection matrix 3x4
    pts_h = np.hstack((pts_3D, np.ones((pts_3D.shape[0],1))))
    proj = P @ pts_h.T
    proj = proj[:2] / proj[2]
    return proj.T.reshape(1, -1, 2).astype(np.float32)

def generate_ratios(n, good_mask):
    #Fake Lowe ratio scores, small for good matches, large for outliers.
    ratios = np.random.uniform(0.1, 0.9, (1, n, 1)).astype(np.float32)
    ratios[0, good_mask, 0] = np.random.uniform(0.1, 0.4, good_mask.sum())
    ratios[0, ~good_mask, 0] = np.random.uniform(0.6, 0.99, (~good_mask).sum())
    return ratios

def generate_synthetic_sample(n_corr=200, outlier_ratio=0.4, noise_std=1.0):
    #Generate one training sample in NG-RANSAC format.
    K1 = random_intrinsics()
    K2 = random_intrinsics()

    R, t = random_pose()
    pts_3D = generate_3D_points(n_corr)

    #  True projections 
    pts1 = project(K1, np.eye(3), np.zeros((3,1)), pts_3D)
    pts2 = project(K2, R, t, pts_3D)

    # Add Gaussian noise
    pts1 += np.random.normal(0, noise_std, pts1.shape)
    pts2 += np.random.normal(0, noise_std, pts2.shape)

    # Add synthetic outliers 
    n = pts1.shape[1]
    n_out = int(outlier_ratio * n)
    out_idx = np.random.choice(n, n_out, replace=False)
    good_mask = np.ones(n, dtype=bool)
    good_mask[out_idx] = False

    pts2[0, out_idx, :] = np.random.uniform(0, 640, (n_out, 2))

    # Lowe ratio meta information
    ratios = generate_ratios(n, good_mask)

    # Image sizes
    im_size1 = np.array([480, 640], dtype=np.float32)
    im_size2 = np.array([480, 640], dtype=np.float32)

    return pts1, pts2, ratios, im_size1, im_size2, K1, K2, R, t


def generate_dataset(n_samples=1000, out_folder="synthetic/"):
    """Generate multiple synthetic files."""
    os.makedirs(out_folder, exist_ok=True)

    for i in range(n_samples):
        sample = generate_synthetic_sample()
        file_path = os.path.join(out_folder, f"{i:06d}.npz")
        np.savez(file_path, *sample)
        print(f"Saved {file_path}")

    print("Done.")

# RUN TO GENERATE DATA 
generate_dataset(n_samples=2000, out_folder="synthetic_train/")
