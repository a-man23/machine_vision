import os
import matplotlib.pyplot as plt
from PIL import Image

def create_collage():
    viz_folder = "viz"
    images = []
    for i in range(15):  # hardcoded from 0 to 14
        img_path = os.path.join(viz_folder, f"viz_pair_{i:02d}.png")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            images.append((i, img))
        else:
            print(f"Warning: {img_path} not found, skipping")
    if not images:
        print("No images found!")
        return
    num_images = len(images)
    num_cols = 3
    num_rows = (num_images + num_cols - 1) // num_cols  # ceiling division

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 2 * num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    for idx, (pair_num, img) in enumerate(images):
        row = idx // num_cols
        col = idx % num_cols
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"Pair {pair_num:02d}", fontsize=12)
        axes[row, col].axis('off')
    for idx in range(num_images, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        axes[row, col].axis('off')
    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)

    output_path = os.path.join(viz_folder, "collage.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Collage saved to {output_path}")

if __name__ == "__main__":
    create_collage()
