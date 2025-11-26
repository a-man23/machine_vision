import cv2
import numpy as np
import os

folder = "images/real"       # your real images folder
out = "real_eval_npz"        # output folder for npz files
os.makedirs(out, exist_ok=True)

sift = cv2.SIFT_create()

# Sort files alphabetically: img1a, img1b, img2a, ...
files = sorted(os.listdir(folder))

# Pair every two files in sorted order
pairs = [(files[i], files[i+1]) for i in range(0, len(files), 2)]

print("Found image pairs:")
for p in pairs:
    print("  ", p)
print("")

for i, (fa, fb) in enumerate(pairs):
    pathA = os.path.join(folder, fa)
    pathB = os.path.join(folder, fb)

    img1 = cv2.imread(pathA, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(pathB, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print(f"Skipping invalid pair {fa}, {fb}")
        continue

    # SIFT feature extraction
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)

    if d1 is None or d2 is None:
        print(f"Skipping pair {fa}, {fb} (no SIFT features found)")
        continue

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(d1, d2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        print(f"Skipping pair {fa}, {fb} (not enough good matches: {len(good)})")
        continue

    pts1 = np.float32([k1[m.queryIdx].pt for m in good])
    pts2 = np.float32([k2[m.trainIdx].pt for m in good])

    np.savez(os.path.join(out, f"pair_{i:02d}.npz"),
             pts1=pts1, pts2=pts2)

    print(f"Saved {len(good)} matches → {os.path.join(out, f'pair_{i:02d}.npz')}")

