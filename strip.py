import os
import numpy as np
from PIL import Image

# --- CONFIG ---
dir_path = r"C:\Users\SWARAJ\OneDrive\Desktop\tpa-new\LMC-Memory\test_results_testsample_modded\video_0_50"  # your folder
output_path = "output_5x5_strip.jpg"
red_line_width = 4     # pixels
grid_rows = 5
grid_cols = 5
frame_step = 2         # take every 2nd image (0, 2, 4, ...)
special_name = "output_input_00009.jpg"  # red line after this frame

# --- SCRIPT ---
# Get sorted image list
img_files = sorted(
    [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
)

# Take every alternate file
img_files = img_files[::frame_step]
img_files = img_files[:grid_rows * grid_cols]  # take first 25 frames max

if not img_files:
    raise RuntimeError('No images found in directory!')

# Load images and normalize
imgs = [Image.open(os.path.join(dir_path, f)).convert('RGB') for f in img_files]
img_size = imgs[0].size
imgs = [im.resize(img_size) if im.size != img_size else im for im in imgs]

# Convert to numpy arrays
img_arrays = [np.array(im) for im in imgs]

# Add red vertical line after the special frame if it appears
for i, fname in enumerate(img_files):
    if fname == special_name:
        red_block = np.zeros((img_size[1], red_line_width, 3), dtype=np.uint8)
        red_block[..., 0] = 255
        img_arrays[i] = np.hstack((img_arrays[i], red_block))

# Build grid (5x5)
rows = []
idx = 0
for r in range(grid_rows):
    row_imgs = []
    for c in range(grid_cols):
        if idx < len(img_arrays):
            row_imgs.append(img_arrays[idx])
            idx += 1
        else:
            # fill with black if not enough images
            row_imgs.append(np.zeros_like(img_arrays[0]))
    row = np.hstack(row_imgs)
    rows.append(row)

# Stack vertically to form full grid
final_image = Image.fromarray(np.vstack(rows))
final_image.save(output_path)

print(f" 5x5 grid saved as {output_path}")
print(f" Using every {frame_step}th image; red line after '{special_name}' if present.")
