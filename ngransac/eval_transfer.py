import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from network import CNNet

def load_npz_pairs(folder):
    pairs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(".npz"):
            data = np.load(os.path.join(folder, f))
            pts1 = data["pts1"].astype(np.float32)
            pts2 = data["pts2"].astype(np.float32)
            pairs.append((f, pts1, pts2))
    return pairs

def epipolar_errors(F, pts1, pts2):
    p1 = np.hstack([pts1, np.ones((len(pts1), 1), dtype=np.float32)])
    p2 = np.hstack([pts2, np.ones((len(pts2), 1), dtype=np.float32)])
    l2 = p1 @ F.T
    denom = np.sqrt(l2[:, 0]**2 + l2[:, 1]**2 + 1e-9)
    d = np.abs(np.sum(l2 * p2, axis=1)) / denom
    return d

def run_ransac(pts1, pts2, thresh=0.5):
    F, mask = cv2.findFundamentalMat(
        pts1, pts2,
        cv2.FM_RANSAC,
        ransacReprojThreshold=thresh
    )
    if F is None:
        return None, None, None
    mask = mask.ravel().astype(bool)
    err = epipolar_errors(F, pts1, pts2)
    return F, mask, err

def make_corr_tensor(pts1, pts2):
    N = len(pts1)
    corr = np.zeros((1, 5, N, 1), dtype=np.float32)
    corr[0, 0, :, 0] = pts1[:, 0]
    corr[0, 1, :, 0] = pts1[:, 1]
    corr[0, 2, :, 0] = pts2[:, 0]
    corr[0, 3, :, 0] = pts2[:, 1]
    corr[0, 4, :, 0] = 1.0
    return torch.from_numpy(corr)


def run_ng_guided_filtered(model, pts1, pts2, keep_ratio=0.5, thresh=0.5):
    N = len(pts1)
    if N < 8:
        return None, None, None, None

    corr = make_corr_tensor(pts1, pts2)
    with torch.no_grad():
        log_probs = model(corr)
        probs = torch.exp(log_probs).view(-1).numpy()

    K = max(8, int(keep_ratio * N))
    idx_sorted = np.argsort(-probs)
    sel = idx_sorted[:K]

    pts1_sel = pts1[sel]
    pts2_sel = pts2[sel]

    F, mask_sel = cv2.findFundamentalMat(
        pts1_sel, pts2_sel,
        cv2.FM_RANSAC,
        ransacReprojThreshold=thresh
    )
    if F is None:
        return None, None, None, probs

    err_all = epipolar_errors(F, pts1, pts2)
    inliers_all = err_all < thresh

    return F, inliers_all, err_all, probs

def get_left_image_path(pair_fname):
    idx_str = pair_fname.split('_')[1].split('.')[0]   # '00'
    idx = int(idx_str)
    return os.path.join("images", "real", f"img{idx}a.jpg")


def visualize_pair_image(fname, pts1, mask_r, mask_ng, probs):
    img_path = get_left_image_path(fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: could not load image {img_path}, skipping image visualization.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    os.makedirs("viz", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.imshow(img_rgb)
    sc = ax.scatter(pts1[:, 0], pts1[:, 1], c=probs, cmap='viridis', s=6)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("NG probability per correspondence")
    ax.axis("off")

    ax = axes[1]
    ax.imshow(img_rgb)
    ax.scatter(pts1[mask_r, 0], pts1[mask_r, 1], s=6, c="lime", label="inliers")
    ax.scatter(pts1[~mask_r, 0], pts1[~mask_r, 1], s=6, c="red", alpha=0.3, label="outliers")
    ax.set_title("RANSAC inliers")
    ax.axis("off")

    ax = axes[2]
    ax.imshow(img_rgb)
    ax.scatter(pts1[mask_ng, 0], pts1[mask_ng, 1], s=6, c="lime", label="inliers")
    ax.scatter(pts1[~mask_ng, 0], pts1[~mask_ng, 1], s=6, c="red", alpha=0.3, label="outliers")
    ax.set_title("NG-guided inliers")
    ax.axis("off")

    out_path = os.path.join("viz", f"viz_{fname.replace('.npz', '.png')}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved image visualization to {out_path}")

def plot_global_stats(pair_names, r_inliers, ng_inliers, r_errs, ng_errs, totals):
    os.makedirs("viz", exist_ok=True)
    num_pairs = len(pair_names)
    x = np.arange(num_pairs)

    r_frac = [ri / t for ri, t in zip(r_inliers, totals)]
    ng_frac = [ni / t for ni, t in zip(ng_inliers, totals)]

    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.bar(x - width/2, r_frac, width, label="RANSAC")
    ax.bar(x + width/2, ng_frac, width, label="NG-guided")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names)
    ax.set_ylabel("Inlier fraction")
    ax.set_title("Inlier fraction per pair")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x - width/2, r_errs, width, label="RANSAC")
    ax.bar(x + width/2, ng_errs, width, label="NG-guided")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names)
    ax.set_ylabel("Median epipolar error (px)")
    ax.set_title("Median epipolar error per pair")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join("viz", "summary_ransac_vs_ng.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved summary stats plot to {out_path}")


def plot_agreement_bars(pair_names, cnt_agree, cnt_r_only, cnt_ng_only, totals):
    os.makedirs("viz", exist_ok=True)
    num_pairs = len(pair_names)
    x = np.arange(num_pairs)

    agree_f   = [a / t for a, t in zip(cnt_agree, totals)]
    r_only_f  = [r / t for r, t in zip(cnt_r_only, totals)]
    ng_only_f = [n / t for n, t in zip(cnt_ng_only, totals)]
    none_f    = [1.0 - (a + r + n) / t for a, r, n, t in zip(cnt_agree, cnt_r_only, cnt_ng_only, totals)]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(x, agree_f, label="inlier for both", color="gold")
    ax.bar(x, r_only_f, bottom=agree_f, label="RANSAC only", color="red", alpha=0.7)
    bottom2 = [a + r for a, r in zip(agree_f, r_only_f)]
    ax.bar(x, ng_only_f, bottom=bottom2, label="NG-guided only", color="limegreen", alpha=0.7)
    bottom3 = [a + r + n for a, r, n in zip(agree_f, r_only_f, ng_only_f)]
    ax.bar(x, none_f, bottom=bottom3, label="neither", color="gray", alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(pair_names)
    ax.set_ylabel("Fraction of correspondences")
    ax.set_title("Inlier agreement / disagreement per pair")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join("viz", "summary_inlier_agreement.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved inlier agreement plot to {out_path}")

def main():
    real_folder = "real_eval_npz"
    model_path = "synthetic_weights_e2e/weights_best.net"

    print("Loading synthetic-trained NG-RANSAC model (guidance network only)...")
    model = CNNet(2)
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()

    print("Loading real correspondence pairs...")
    pairs = load_npz_pairs(real_folder)
    print(f"Found {len(pairs)} real image pairs.\n")

    keep_ratio = 0.5


    pair_names = []
    r_inliers = []
    ng_inliers = []
    r_errs = []
    ng_errs = []
    totals = []
    cnt_agree = []
    cnt_r_only = []
    cnt_ng_only = []

    for fname, pts1, pts2 in pairs:
        print(f"Evaluating pair: {fname}")

        total = len(pts1)

        Fr, mask_r, err_r = run_ransac(pts1, pts2)
        if Fr is None:
            print("RANSAC failed.\n")
            continue

        print(f"RANSAC:      inliers = {mask_r.sum()} / {len(mask_r)}")
        print(f"RANSAC:      median epipolar error = {np.median(err_r):.4f}")

        Fng, mask_ng, err_ng, probs = run_ng_guided_filtered(
            model, pts1, pts2, keep_ratio=keep_ratio, thresh=0.5
        )

        if Fng is None:
            print("NG-guided RANSAC failed.\n")
            continue

        print(f"NG-guided:   inliers = {mask_ng.sum()} / {len(mask_ng)}")
        print(f"NG-guided:   median epipolar error = {np.median(err_ng):.4f}")
        print("")

        pair_id = fname.replace(".npz", "")  # e.g. 'pair_00'
        pair_names.append(pair_id)
        r_inliers.append(int(mask_r.sum()))
        ng_inliers.append(int(mask_ng.sum()))
        r_errs.append(float(np.median(err_r)))
        ng_errs.append(float(np.median(err_ng)))
        totals.append(total)

        agree   = mask_r & mask_ng
        r_only  = mask_r & ~mask_ng
        ng_only = ~mask_r & mask_ng

        cnt_agree.append(int(agree.sum()))
        cnt_r_only.append(int(r_only.sum()))
        cnt_ng_only.append(int(ng_only.sum()))

        visualize_pair_image(fname, pts1, mask_r, mask_ng, probs)

    if pair_names:
        plot_global_stats(pair_names, r_inliers, ng_inliers, r_errs, ng_errs, totals)
        plot_agreement_bars(pair_names, cnt_agree, cnt_r_only, cnt_ng_only, totals)


if __name__ == "__main__":
    main()