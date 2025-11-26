import numpy as np
import cv2
import os

def random_intrinsics(fx=800, fy=800, cx=320, cy=240):
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    return K

def random_pose(max_angle_deg=15, max_translation=0.1):
    angle = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg, 3))
    Rx = cv2.Rodrigues(np.array([angle[0],0,0]))[0]
    Ry = cv2.Rodrigues(np.array([0,angle[1],0]))[0]
    Rz = cv2.Rodrigues(np.array([0,0,angle[2]]))[0]
    R = Rz @ Ry @ Rx

    t = np.random.uniform(-max_translation, max_translation, 3)
    t = t / np.linalg.norm(t)
    t = t.reshape(3,1)
    return R.astype(np.float32), t.astype(np.float32)

def generate_3D_points(n):
    pts = np.random.uniform(-0.5,0.5,(n,3))
    pts[:,2] += 3.0
    return pts

def project(K,R,t,pts_3D):
    P = K @ np.hstack((R,t))
    pts_h = np.hstack((pts_3D,np.ones((pts_3D.shape[0],1))))
    proj = P @ pts_h.T
    proj = proj[:2] / proj[2]
    return proj.T.reshape(1,-1,2).astype(np.float32)

def generate_ratios(n, good_mask):
    ratios = np.zeros((1,n,1), dtype=np.float32)
    ratios[0, good_mask, 0] = np.random.uniform(0.1, 0.5, good_mask.sum())
    # Outliers: mostly high ratios but some small values (long tail)
    # Beta(a, b) with a < b skews toward 0; flip to skew toward 1
    beta_samples = np.random.beta(a=0.8, b=5, size=(~good_mask).sum())  # skew toward 0
    beta_samples = 1.0 - beta_samples  # now skewed toward 1
    ratios[0, ~good_mask, 0] = beta_samples  # no clipping, can go down toward 0
    return ratios

def generate_synthetic_sample(n_corr=3000, outlier_ratio=0.9, noise_std=1):
    K1 = random_intrinsics()
    K2 = random_intrinsics()
    R,t = random_pose()
    pts_3D = generate_3D_points(n_corr)

    pts1 = project(K1,np.eye(3),np.zeros((3,1)),pts_3D)
    pts2 = project(K2,R,t,pts_3D)

    pts1 += np.random.normal(0,noise_std,pts1.shape)
    pts2 += np.random.normal(0,noise_std,pts2.shape)

    n = pts1.shape[1]
    n_out = int(outlier_ratio*n)
    out_idx = np.random.choice(n,n_out,replace=False)
    good_mask = np.ones(n,dtype=bool)
    good_mask[out_idx] = False

    # Split outliers into near and far
    n_near = int(0.6 * n_out)
    n_far = n_out - n_near
    out_indices = out_idx
    near_idx = np.random.choice(out_indices, n_near, replace=False)
    far_idx = np.setdiff1d(out_indices, near_idx)

    # Near outliers: small random offset from correct projection
    pts2[0, near_idx, :] += np.random.normal(0, 20, (n_near,2))

    # Far outliers: completely random in image
    pts2[0, far_idx, :] = np.random.uniform(0, 640, (n_far,2))

    # Generate ratios
    ratios = generate_ratios(n, good_mask)

    im_size1 = np.array([480,640],dtype=np.float32)
    im_size2 = np.array([480,640],dtype=np.float32)

    return pts1, pts2, ratios, im_size1, im_size2, K1, K2, R, t

def generate_dataset(n_samples=1000,out_folder="synthetic_train2/"):
    os.makedirs(out_folder,exist_ok=True)
    for i in range(n_samples):
        sample = generate_synthetic_sample()
        file_path = os.path.join(out_folder,f"{i:06d}.npz")
        np.savez(file_path,*sample)
        if i % 100 == 0:
            print(f"Saved {file_path}")
    print("Done.")

if __name__ == "__main__":
    generate_dataset(n_samples=10000,out_folder="synthetic_train2/")