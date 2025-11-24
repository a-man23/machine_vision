import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def parse_train_log(log_file='synthetic_weights_e2e/train_log.txt'):
	"""Parse train_log.txt to extract train and validation losses."""
	train_losses_iter = []
	train_losses_epoch = []
	val_losses_epoch = []
	epochs = []
	
	with open(log_file, 'r') as f:
		for line in f:
			line = line.strip()
			if line.startswith('iter'):
				parts = line.split()
				if len(parts) >= 3:
					train_losses_iter.append(float(parts[2]))
			elif line.startswith('epoch'):
				parts = line.split()
				if len(parts) >= 4:
					epochs.append(int(parts[1]))
					train_losses_epoch.append(float(parts[2]))
					val_losses_epoch.append(float(parts[3]))
	
	return train_losses_iter, train_losses_epoch, val_losses_epoch, epochs

def plot_training_curves(log_file, save_path=None):
	"""Plot training and validation loss curves."""
	train_losses_iter, train_losses_epoch, val_losses_epoch, epochs = parse_train_log(log_file)
	
	if len(epochs) == 0:
		print("No epoch data found in log file!")
		return
	
	# Create figure with two subplots
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
	
	# Plot 1: Iteration-level training loss
	if len(train_losses_iter) > 0:
		iterations = np.arange(len(train_losses_iter))
		ax1.plot(iterations, train_losses_iter, 'b-', alpha=0.3, linewidth=0.5, label='Train Loss (per iteration)')
		# Also plot smoothed version
		if len(train_losses_iter) > 100:
			window = min(100, len(train_losses_iter) // 10)
			smoothed = np.convolve(train_losses_iter, np.ones(window)/window, mode='valid')
			ax1.plot(iterations[window-1:], smoothed, 'b-', linewidth=1.5, label=f'Train Loss (smoothed, window={window})')
		else:
			ax1.plot(iterations, train_losses_iter, 'b-', linewidth=1.5, label='Train Loss')
	
	ax1.set_xlabel('Iteration')
	ax1.set_ylabel('Loss')
	ax1.set_title('Training Loss (per iteration)')
	ax1.legend()
	ax1.grid(True, alpha=0.3)
	ax1.invert_yaxis()  # Since loss is negative (more negative = better)
	
	# Plot 2: Epoch-level train and validation loss
	ax2.plot(epochs, train_losses_epoch, 'b-o', label='Train Loss', markersize=6, linewidth=2)
	ax2.plot(epochs, val_losses_epoch, 'r-s', label='Validation Loss', markersize=6, linewidth=2)
	
	ax2.set_xlabel('Epoch')
	ax2.set_ylabel('Loss')
	ax2.set_title('Training and Validation Loss (per epoch)')
	ax2.legend()
	ax2.grid(True, alpha=0.3)
	ax2.invert_yaxis()  # Since loss is negative (more negative = better)
	
	# Add inlier rate on secondary y-axis for epoch plot
	ax2_twin = ax2.twinx()
	train_inlier_rates = [-x * 100 for x in train_losses_epoch]
	val_inlier_rates = [-x * 100 for x in val_losses_epoch]
	ax2_twin.plot(epochs, train_inlier_rates, 'b--', alpha=0.5, linewidth=1)
	ax2_twin.plot(epochs, val_inlier_rates, 'r--', alpha=0.5, linewidth=1)
	ax2_twin.set_ylabel('Inlier Rate (%)', color='gray')
	ax2_twin.tick_params(axis='y', labelcolor='gray')
	
	plt.tight_layout()
	
	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"Plot saved to {save_path}")
	else:
		plt.show()
	
	return fig

if __name__ == '__main__':
	# Default log file path
	default_log_file = 'synthetic_weights_e2e/train_log.txt'
	
	if len(sys.argv) < 2:
		# Use default path if no arguments provided
		log_file = default_log_file
		print(f"Using default log file: {log_file}")
	else:
		log_file = sys.argv[1]
	
	if not os.path.exists(log_file):
		print(f"Error: Log file '{log_file}' not found!")
		print(f"Usage: python plot_training_curves.py [train_log.txt] [output.png]")
		sys.exit(1)
	
	output_path = sys.argv[2] if len(sys.argv) > 2 else None
	plot_training_curves(log_file, output_path)

