import numpy as np
import os
import matplotlib.pyplot as plt

DATA_FOLDER = "synthetic_train2/"  # your dataset folder
MAX_POINTS_PLOT = 50
MIN_OUTLIER_RATIO = 0.05  # Warn if fewer than 5% outliers

def check_sample(file_path, max_points=MAX_POINTS_PLOT):
    data = np.load(file_path)

    # Access arrays by arr_0, arr_1, etc.
    pts1 = data["arr_0"]  # (1, n_points, 2)
    pts2 = data["arr_1"]
    ratios = data["arr_2"]  # (1, n_points, 1)

    n_points = pts1.shape[1]

    # Simple threshold to classify inliers/outliers
    inlier_mask = (ratios <= 0.5).reshape(-1)
    outlier_mask = ~inlier_mask

    n_inliers = np.sum(inlier_mask)
    n_outliers = np.sum(outlier_mask)
    outlier_ratio = n_outliers / n_points

    print(f"\n{os.path.basename(file_path)}: total={n_points}, inliers={n_inliers}, outliers={n_outliers} ({outlier_ratio:.2f})")
    if outlier_ratio < MIN_OUTLIER_RATIO:
        print("⚠ Warning: Low outlier ratio, may confuse NG-RANSAC")

    # -----------------------------
    # 1. Distance histogram
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
    # 2. Ratio histogram
    # -----------------------------
    plt.figure(figsize=(6,4))
    plt.hist(ratios.reshape(-1)[inlier_mask], bins=20, alpha=0.6, label="inliers")
    plt.hist(ratios.reshape(-1)[outlier_mask], bins=20, alpha=0.6, label="outliers")
    plt.xlabel("Lowe ratio")
    plt.ylabel("Count")
    plt.title("Ratio distribution")
    plt.legend()
    plt.show()

    # -----------------------------
    # 3. Point scatter + lines
    # -----------------------------
    pts_to_plot = min(max_points, n_points)
    plt.figure(figsize=(7,7))

    # Plot inliers
    plt.scatter(pts1[0,:pts_to_plot][inlier_mask[:pts_to_plot],0],
                pts1[0,:pts_to_plot][inlier_mask[:pts_to_plot],1],
                c='g', label='pts1 inlier', alpha=0.6)
    plt.scatter(pts2[0,:pts_to_plot][inlier_mask[:pts_to_plot],0],
                pts2[0,:pts_to_plot][inlier_mask[:pts_to_plot],1],
                c='lime', label='pts2 inlier', alpha=0.6)

    # Plot outliers
    plt.scatter(pts1[0,:pts_to_plot][outlier_mask[:pts_to_plot],0],
                pts1[0,:pts_to_plot][outlier_mask[:pts_to_plot],1],
                c='r', label='pts1 outlier', alpha=0.6)
    plt.scatter(pts2[0,:pts_to_plot][outlier_mask[:pts_to_plot],0],
                pts2[0,:pts_to_plot][outlier_mask[:pts_to_plot],1],
                c='orange', label='pts2 outlier', alpha=0.6)

    # Draw lines connecting correspondences (color by distance)
    for i in range(pts_to_plot):
        dist = np.linalg.norm(pts1[0,i] - pts2[0,i])
        color = 'g' if inlier_mask[i] else 'r'
        plt.plot([pts1[0,i,0], pts2[0,i,0]],
                 [pts1[0,i,1], pts2[0,i,1]],
                 c=color, alpha=min(0.5, dist/200))

    plt.legend()
    plt.title(f"{os.path.basename(file_path)} - first {pts_to_plot} points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().invert_yaxis()  # match image coordinates
    plt.show()


def check_dataset(folder=DATA_FOLDER, n_samples=5):
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npz")])
    print(f"Found {len(files)} samples. Checking first {n_samples} samples...")
    
    for file_path in files[:n_samples]:
        check_sample(file_path)


if __name__ == "__main__":
    check_dataset()