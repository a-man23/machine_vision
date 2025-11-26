import os
os.add_dll_directory(r"C:\Users\Aman.DESKTOP-VEB4T1B\AppData\Local\Programs\Python\Python312\Lib\site-packages\opencv\build\x64\vc16\bin")
import numpy as np
import cv2
import random

import torch
import torch.optim as optim
import ngransac

from network import CNNet
from dataset_synthetic import SparseDatasetSynthetic
import util

def collate_fn_pad(batch):
	"""
	Custom collate function to handle variable-length correspondences.
	Pads correspondences to the maximum length in the batch.
	"""
	correspondences_list = []
	gt_F_list = []
	gt_E_list = []
	gt_R_list = []
	gt_t_list = []
	K1_list = []
	K2_list = []
	im_size1_list = []
	im_size2_list = []
	
	# Find maximum number of correspondences in the batch
	max_corrs = max(item[0].shape[1] for item in batch)
	
	for correspondences, gt_F, gt_E, gt_R, gt_t, K1, K2, im_size1, im_size2 in batch:
		# correspondences shape: (5, N, 1)
		num_corrs = correspondences.shape[1]
		
		# Pad correspondences to max_corrs
		if num_corrs < max_corrs:
			# Pad with zeros along dimension 1
			padding = torch.zeros(5, max_corrs - num_corrs, 1)
			correspondences = torch.cat([correspondences, padding], dim=1)
		
		correspondences_list.append(correspondences)
		gt_F_list.append(gt_F)
		gt_E_list.append(gt_E)
		gt_R_list.append(gt_R)
		gt_t_list.append(gt_t)
		K1_list.append(K1)
		K2_list.append(K2)
		im_size1_list.append(im_size1)
		im_size2_list.append(im_size2)
	
	# Stack all tensors
	correspondences = torch.stack(correspondences_list, dim=0)
	gt_F = torch.stack(gt_F_list, dim=0)
	gt_E = torch.stack(gt_E_list, dim=0)
	gt_R = torch.stack(gt_R_list, dim=0)
	gt_t = torch.stack(gt_t_list, dim=0)
	K1 = torch.stack(K1_list, dim=0)
	K2 = torch.stack(K2_list, dim=0)
	im_size1 = torch.stack(im_size1_list, dim=0)
	im_size2 = torch.stack(im_size2_list, dim=0)
	
	return correspondences, gt_F, gt_E, gt_R, gt_t, K1, K2, im_size1, im_size2

# Configuration
synthetic_data_folder = "synthetic_train2"
output_weights_folder = "synthetic_weights_e2e_2"
os.makedirs(output_weights_folder, exist_ok=True)

# Training parameters
epochs = 50
batch_size = 8
learning_rate = 0.0001  # Increased from 0.00001 for faster learning
resblocks = 2
hyps = 16
sample_count = 4
loss_type = 'inliers'  # 'pose', 'inliers', 'f1', 'epi' - using inliers since pose recovery is failing
ratio_threshold = 1.0
nfeatures = -1
threshold = 0.001
init_model = ''  # Path to initialization model if available

# Early stopping parameters
early_stopping_patience = 5  # Number of epochs to wait before stopping
early_stopping_min_delta = 0.0001  # Minimum change to qualify as an improvement
early_stopping_enabled = True

# Create full list of .npz files
all_npz_files = [os.path.join(synthetic_data_folder, f) for f in sorted(os.listdir(synthetic_data_folder)) if f.endswith('.npz')]

# Randomly select 2000 files (or all if fewer than 2000 available)
num_files_to_select = min(2000, len(all_npz_files))
npz_files = random.sample(all_npz_files, num_files_to_select)

print(f'Total .npz files available: {len(all_npz_files)}')
print(f'Randomly selected: {len(npz_files)} files')

# Split into train/val (90/10 split)
split_idx = int(0.9 * len(npz_files))
train_files = npz_files[:split_idx]
val_files = npz_files[split_idx:]

print('Using synthetic dataset:')
print(f'  Training samples: {len(train_files)}')
print(f'  Validation samples: {len(val_files)}')

# Create datasets
trainset = SparseDatasetSynthetic([train_files], ratio_threshold, nfeatures, fmat=False, overwrite_side_info=False)
valset = SparseDatasetSynthetic([val_files], ratio_threshold, nfeatures, fmat=False, overwrite_side_info=False)

trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=0, batch_size=batch_size, collate_fn=collate_fn_pad)
valset_loader = torch.utils.data.DataLoader(valset, shuffle=False, num_workers=0, batch_size=batch_size, collate_fn=collate_fn_pad)

print(f"\nImage pairs: {len(trainset)}\n")

# create or load model
model = CNNet(resblocks)
if len(init_model) > 0 and os.path.exists(init_model):
	print(f"Loading model from {init_model}")
	model.load_state_dict(torch.load(init_model))
model = model.cuda()
model.train()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

iteration = 0
train_losses = []
val_losses = []

# Early stopping tracking
best_val_loss = float('inf')
patience_counter = 0
best_epoch = 0

# Training log file
train_log = open(os.path.join(output_weights_folder, 'train_log.txt'), 'w', 1)

# main training loop
for epoch in range(0, epochs):	

	print("=== Starting Epoch", epoch, "==================================")

	# store the network every epoch
	torch.save(model.state_dict(), os.path.join(output_weights_folder, f'weights_epoch_{epoch}.net'))

	epoch_train_loss = 0.0
	num_batches = 0

	# main training loop in the current epoch
	for correspondences, gt_F, gt_E, gt_R, gt_t, K1, K2, im_size1, im_size2 in trainset_loader:

		gt_R = gt_R.numpy()
		gt_t = gt_t.numpy()

		# predict neural guidance
		log_probs = model(correspondences.cuda())
		probs = torch.exp(log_probs).cpu()

		# this tensor will contain the gradients for the entire batch
		log_probs_grad = torch.zeros(log_probs.size())

		avg_loss = 0

		#loop over batch
		for b in range(correspondences.size(0)):

			# we sample multiple times per input and keep the gradients and losses in the following lists
			log_prob_grads = [] 
			losses = []

			# loop over samples for approximating the expected loss
			for s in range(sample_count):

				# gradient tensor of the current sample
				# when running NG-RANSAC, this tensor will indicate which correspondences have been sampled
				# this is multiplied with the loss of the sample to yield the gradients for log-probabilities
				gradients = torch.zeros(probs[b].size()) 

				 # inlier mask of the best model
				inliers = torch.zeros(probs[b].size())

				# random seed used in C++ (would be initialized in each call with the same seed if not provided from outside)
				rand_seed = random.randint(0, 10000) 

				# === CASE ESSENTIAL MATRIX =========================================

				# run NG-RANSAC
				E = torch.zeros((3, 3)).float()
				incount = ngransac.find_essential_mat(correspondences[b], probs[b], rand_seed, hyps, threshold, E, inliers, gradients)		
				incount /= correspondences.size(2)

				# Extract points from correspondences
				# correspondences[b] is (5, N, 1) where first 2 are pts1, next 2 are pts2, last is ratio
				# correspondences format: (5, N, 1) -> after indexing [b,0:2] -> (2, N, 1) -> squeeze -> (2, N) -> T -> (N, 2)
				corr_b = correspondences[b]  # (5, N, 1)
				pts1 = corr_b[0:2].squeeze().numpy().T  # (N, 2)
				pts2 = corr_b[2:4].squeeze().numpy().T  # (N, 2)
				
				# Ensure we have valid points
				if pts1.shape[0] == 0 or pts2.shape[0] == 0:
					loss = -incount
					log_prob_grads.append(gradients)
					losses.append(loss)
					continue

				# choose the user-defined training signal
				if loss_type == 'inliers':
					loss = -incount
				elif loss_type == 'f1' and False:  # f1 only for fundamental matrix
					loss = -F1
				elif loss_type == 'epi' and False:  # epi only for fundamental matrix
					loss = epi_error
				else:
					# evaluation of relative pose (essential matrix)
					# Points are already in normalized coordinates (from dataset)
					# cv2.recoverPose expects points in shape (N, 2) with float64, contiguous
					pts1_np = np.ascontiguousarray(pts1, dtype=np.float64)
					pts2_np = np.ascontiguousarray(pts2, dtype=np.float64)
					
					# Ensure inliers mask is correct format - must be 1D array matching number of points
					inliers_np = inliers.byte().numpy()
					if inliers_np.ndim > 1:
						inliers_np = inliers_np.squeeze()
					if inliers_np.ndim > 1:
						inliers_np = inliers_np.ravel()
					inliers_np = np.ascontiguousarray(inliers_np, dtype=np.uint8)
					
					# Ensure inliers length matches points
					if len(inliers_np) != len(pts1_np):
						# Truncate or pad inliers to match
						if len(inliers_np) > len(pts1_np):
							inliers_np = inliers_np[:len(pts1_np)]
						else:
							# Pad with zeros
							padding = np.zeros(len(pts1_np) - len(inliers_np), dtype=np.uint8)
							inliers_np = np.concatenate([inliers_np, padding])
					
					E_np = E.double().numpy()
					K = np.eye(3)
					R = np.eye(3)
					t = np.zeros((3,1))

					# For now, use inlier count loss since pose recovery is having issues
					# TODO: Fix pose recovery - points might need different format
					loss = -incount
					
					# Uncomment below to try pose recovery (currently disabled due to OpenCV issues)
					# try:
					# 	if pts1_np.shape != pts2_np.shape or pts1_np.shape[1] != 2:
					# 		raise ValueError(f"Invalid point shapes")
					# 	if len(inliers_np) != len(pts1_np):
					# 		raise ValueError(f"Inliers length mismatch")
					# 	cv2.recoverPose(E_np, pts1_np, pts2_np, K, R, t, inliers_np)
					# 	dR, dT = util.pose_error(R, gt_R[b], t, gt_t[b])
					# 	loss = max(float(dR), float(dT))
					# except:
					# 	loss = -incount
						
				log_prob_grads.append(gradients)
				losses.append(loss)
		
			# calculate the gradients of the expected loss
			baseline = sum(losses) / len(losses) #expected loss
			for i, l in enumerate(losses): # subtract baseline for each sample to reduce gradient variance
				log_probs_grad[b] += log_prob_grads[i] * (l - baseline) / sample_count

			avg_loss += baseline

		avg_loss /= correspondences.size(0)

		# Log training loss per iteration
		train_log.write('iter %d %f\n' % (iteration, avg_loss))

		# update model
		torch.autograd.backward((log_probs), (log_probs_grad.cuda()))
		optimizer.step() 
		optimizer.zero_grad()

		if iteration % 10 == 0:
			# Loss is negative because we're maximizing inliers: loss = -inlier_count
			# More negative = better (more inliers found)
			inlier_rate = -avg_loss  # Convert back to inlier rate for readability
			print(f"Iteration: {iteration:4d} | Loss: {avg_loss:7.5f} | Inlier Rate: {inlier_rate*100:5.2f}%")

		epoch_train_loss += avg_loss
		num_batches += 1
		iteration += 1

	# Average training loss for this epoch
	epoch_train_loss = epoch_train_loss / num_batches if num_batches > 0 else 0.0
	train_losses.append(epoch_train_loss)

	# Validation
	model.eval()
	val_loss = 0.0
	val_iter = 0
	with torch.no_grad():
		for correspondences, gt_F, gt_E, gt_R, gt_t, K1, K2, im_size1, im_size2 in valset_loader:
			gt_R = gt_R.numpy()
			gt_t = gt_t.numpy()
			
			log_probs = model(correspondences.cuda())
			probs = torch.exp(log_probs).cpu()
			
			batch_val_loss = 0.0
			for b in range(correspondences.size(0)):
				gradients = torch.zeros(probs[b].size())
				inliers = torch.zeros(probs[b].size())
				rand_seed = random.randint(0, 10000)
				
				E = torch.zeros((3, 3)).float()
				incount = ngransac.find_essential_mat(correspondences[b], probs[b], rand_seed, hyps, threshold, E, inliers, gradients)
				incount /= correspondences.size(2)
				
				# Use inlier count loss (pose recovery disabled)
				loss = -incount
				
				batch_val_loss += loss
			
			val_loss += batch_val_loss / correspondences.size(0)
			val_iter += 1
	
	avg_val_loss = val_loss / val_iter if val_iter > 0 else 0.0
	val_losses.append(avg_val_loss)
	
	# Early stopping check
	# Note: Since loss is negative (more negative = better), we check if val_loss < best_val_loss
	improved = False
	if avg_val_loss < best_val_loss - early_stopping_min_delta:
		best_val_loss = avg_val_loss
		best_epoch = epoch
		patience_counter = 0
		improved = True
		# Save best model
		torch.save(model.state_dict(), os.path.join(output_weights_folder, 'weights_best.net'))
		print(f"  *** New best validation loss! Saving model...")
	else:
		patience_counter += 1
	
	# Log epoch-level train and validation loss
	train_log.write('epoch %d %f %f\n' % (epoch, epoch_train_loss, avg_val_loss))
	train_log.flush()  # Ensure it's written immediately
	
	# Convert losses to inlier rates for readability
	train_inlier_rate = -epoch_train_loss
	val_inlier_rate = -avg_val_loss
	improvement_str = " [BEST]" if improved else f" [patience: {patience_counter}/{early_stopping_patience}]"
	print(f"Epoch {epoch}: Train Loss: {epoch_train_loss:.4f} (Inlier Rate: {train_inlier_rate*100:.2f}%) | "
		  f"Val Loss: {avg_val_loss:.4f} (Inlier Rate: {val_inlier_rate*100:.2f}%){improvement_str}")
	
	# Early stopping check
	if early_stopping_enabled and patience_counter >= early_stopping_patience:
		print(f"\nEarly stopping triggered! No improvement for {early_stopping_patience} epochs.")
		print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
		break

# Save final model
torch.save(model.state_dict(), os.path.join(output_weights_folder, 'weights_final.net'))
train_log.close()

# Save loss arrays
np.save(os.path.join(output_weights_folder, 'train_losses.npy'), np.array(train_losses))
np.save(os.path.join(output_weights_folder, 'val_losses.npy'), np.array(val_losses))

print(f"\nTraining complete!")
print(f"  Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
print(f"  Models saved to {output_weights_folder}/")
print(f"    - weights_best.net: Best model (epoch {best_epoch})")
print(f"    - weights_final.net: Final model (epoch {epoch})")
print(f"    - weights_epoch_*.net: All epoch checkpoints")
