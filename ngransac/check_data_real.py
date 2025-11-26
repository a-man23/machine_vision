import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_scipy
import cv2

DATA_FOLDER = "traindata/kitti/test_data/"
MAX_POINTS_PLOT = 50
MIN_OUTLIER_RATIO = 0.05
REPROJ_THRESHOLD = 2.0  # pixels

def check_pair(file_path, max_points=MAX_POINTS_PLOT):
    data = np.load(file_path, allow_pickle=True)

    pts1 = data[0].astype(np.float32)
    pts2 = data[1].astype(np.float32)
    ratios = data[2].astype(np.float32)  # keep Lowe ratios

    R_gt = data[-2] if data.shape[0] >= 9 else None
    t_gt = data[-1] if data.shape[0] >= 9 else None

    n_points = pts1.shape[1]

    if R_gt is not None and t_gt is not None:
        # Ground-truth info
        r = R_scipy.from_matrix(R_gt)
        angles = r.as_euler('xyz', degrees=True)
        t_norm = np.linalg.norm(t_gt)
        print(f"Ground-truth rotation (deg): {angles}")
        print(f"Ground-truth translation magnitude: {t_norm:.4f}")

        # Build projection matrices
        K1 = np.eye(3, dtype=np.float32)
        K2 = np.eye(3, dtype=np.float32)
        P1 = K1 @ np.hstack([np.eye(3), np.zeros((3,1))])
        P2 = K2 @ np.hstack([R_gt, t_gt.reshape(3,1)])

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
    else:
        # fallback
        inlier_mask = (ratios <= 0.5).reshape(-1)
        outlier_mask = ~inlier_mask

    n_inliers = np.sum(inlier_mask)
    n_outliers = np.sum(outlier_mask)
    outlier_ratio = n_outliers / n_points

    print(f"\n{os.path.basename(file_path)}: total={n_points}, inliers={n_inliers}, "
          f"outliers={n_outliers} ({outlier_ratio:.2f})")
    if outlier_ratio < MIN_OUTLIER_RATIO:
        print("⚠ Warning: Low outlier ratio, may confuse NG-RANSAC")

    # Histogram: 2D distances
    dists = np.linalg.norm(pts1 - pts2, axis=2).reshape(-1)
    plt.figure(figsize=(6,4))
    plt.hist(dists[inlier_mask], bins=20, alpha=0.6, label="inliers")
    plt.hist(dists[outlier_mask], bins=20, alpha=0.6, label="outliers")
    plt.xlabel("2D distance between pts1 and pts2")
    plt.ylabel("Number of points")
    plt.title("Distance distribution")
    plt.legend()
    plt.show()

    # Histogram: Lowe ratios (still shown for all points)
    plt.figure(figsize=(6,4))
    plt.hist(ratios.reshape(-1)[inlier_mask], bins=20, alpha=0.6, label="inliers (reproj)")
    plt.hist(ratios.reshape(-1)[outlier_mask], bins=20, alpha=0.6, label="outliers (reproj)")
    plt.xlabel("Lowe ratio")
    plt.ylabel("Count")
    plt.title("Lowe ratio distribution by reprojection inliers")
    plt.legend()
    plt.show()

    # Scatter plot
    pts_to_plot = min(max_points, n_points)
    plt.figure(figsize=(7,7))
    plt.scatter(pts1[0,:pts_to_plot][inlier_mask[:pts_to_plot],0],
                pts1[0,:pts_to_plot][inlier_mask[:pts_to_plot],1],
                c='g', label='pts1 inlier', alpha=0.6)
    plt.scatter(pts2[0,:pts_to_plot][inlier_mask[:pts_to_plot],0],
                pts2[0,:pts_to_plot][inlier_mask[:pts_to_plot],1],
                c='lime', label='pts2 inlier', alpha=0.6)

    plt.scatter(pts1[0,:pts_to_plot][outlier_mask[:pts_to_plot],0],
                pts1[0,:pts_to_plot][outlier_mask[:pts_to_plot],1],
                c='r', label='pts1 outlier', alpha=0.6)
    plt.scatter(pts2[0,:pts_to_plot][outlier_mask[:pts_to_plot],0],
                pts2[0,:pts_to_plot][outlier_mask[:pts_to_plot],1],
                c='orange', label='pts2 outlier', alpha=0.6)

    for i in range(pts_to_plot):
        color = 'g' if inlier_mask[i] else 'r'
        plt.plot([pts1[0,i,0], pts2[0,i,0]],
                 [pts1[0,i,1], pts2[0,i,1]],
                 c=color, alpha=0.5)

    plt.legend()
    plt.title(f"{os.path.basename(file_path)} - first {pts_to_plot} points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().invert_yaxis()
    plt.show()


def check_dataset(folder=DATA_FOLDER, n_samples=5):
    files = sorted([os.path.join(folder,f) for f in os.listdir(folder) if f.endswith(".npy")])
    print(f"Found {len(files)} samples. Checking first {n_samples} samples...")
    for file_path in files[-n_samples:]:
        check_pair(file_path)


if __name__ == "__main__":
    check_dataset()