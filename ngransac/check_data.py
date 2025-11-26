import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

DATA_FOLDER = "synthetic_train2/"
MAX_POINTS_PLOT = 50
REPROJ_THRESHOLD = 1.0  # pixels

def check_synthetic_sample(file_path, max_points=MAX_POINTS_PLOT):
    data = np.load(file_path)

    pts1 = data["arr_0"].astype(np.float32)
    pts2 = data["arr_1"].astype(np.float32)
    ratios = data["arr_2"].astype(np.float32)

    K1 = data["arr_5"]
    K2 = data["arr_6"]
    R_gt = data["arr_7"]
    t_gt = data["arr_8"].reshape(3,1)

    n_points = pts1.shape[1]

    # Build projection matrices
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3,1))])
    P2 = K2 @ np.hstack([R_gt, t_gt])

    # Triangulate points
    pts_3D_h = cv2.triangulatePoints(P1, P2, pts1[0].T, pts2[0].T)
    pts_3D = (pts_3D_h[:3, :] / pts_3D_h[3, :]).T

    # Reproject
    def reproject(P, pts3D):
        pts_h = np.hstack([pts3D, np.ones((pts3D.shape[0],1))])
        proj = (P @ pts_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        return proj

    proj1 = reproject(P1, pts_3D)
    proj2 = reproject(P2, pts_3D)

    reproj_err = np.maximum(np.linalg.norm(proj1 - pts1[0], axis=1),
                             np.linalg.norm(proj2 - pts2[0], axis=1))
    inlier_mask = reproj_err < REPROJ_THRESHOLD
    outlier_mask = ~inlier_mask

    n_inliers = np.sum(inlier_mask)
    n_outliers = np.sum(outlier_mask)
    outlier_ratio = n_outliers / n_points

    print(f"\n{os.path.basename(file_path)}: total={n_points}, inliers={n_inliers}, "
          f"outliers={n_outliers} ({outlier_ratio:.2f})")

    # -----------------------------
    # Distance histogram
    # -----------------------------
    dists = np.linalg.norm(pts1 - pts2, axis=2).reshape(-1)
    plt.figure(figsize=(6,4))
    plt.hist(dists[inlier_mask], bins=20, alpha=0.6, label="inliers")
    plt.hist(dists[outlier_mask], bins=20, alpha=0.6, label="outliers")
    plt.xlabel("2D distance between pts1 and pts2")
    plt.ylabel("Number of points")
    plt.title("Distance distribution")
    plt.legend()
    plt.show()

    # -----------------------------
    # Ratio histogram
    # -----------------------------
    plt.figure(figsize=(6,4))
    plt.hist(ratios.reshape(-1)[inlier_mask], bins=20, alpha=0.6, label="inliers")
    plt.hist(ratios.reshape(-1)[outlier_mask], bins=20, alpha=0.6, label="outliers")
    plt.xlabel("Lowe ratio")
    plt.ylabel("Count")
    plt.title("Lowe ratio distribution by reprojection inliers")
    plt.legend()
    plt.show()

def check_synthetic_dataset(folder=DATA_FOLDER, n_samples=5):
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npz")])
    print(f"Found {len(files)} samples. Checking first {n_samples} samples...")
    for file_path in files[:n_samples]:
        check_synthetic_sample(file_path)

if __name__ == "__main__":
    check_synthetic_dataset()