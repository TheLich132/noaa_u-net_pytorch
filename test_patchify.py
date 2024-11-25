import cv2
import numpy as np
from patchify import patchify, unpatchify
import matplotlib.pyplot as plt

img = cv2.imread('/var/home/jakub/Muzyka/gqrx/11.06.2024_09:24/raw_sync.png', cv2.IMREAD_GRAYSCALE)
patches = patchify(img, (256, 256), step=256)
print(patches.shape)
patch = patches[0, 0]
print(patch.shape)
patch = patch.astype('float32') / 255.0

# Define the patch size and overlap
patch_size = 256
overlap = 128

# Calculate the step size
step_size = patch_size - overlap

# Create a list to hold the output patches
output_patches = []

# Create a grid of patch positions
for i in range(0, img.shape[0], step_size):
    for j in range(0, img.shape[1], step_size):
        # Extract the patch from the image
        patch = img[max(0, i):min(img.shape[0], i+patch_size), max(0, j):min(img.shape[1], j+patch_size)]

        # Pad the patch to the full patch size if necessary
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            padded_patch = np.zeros((patch_size, patch_size), dtype=np.uint8)
            padded_patch[:patch.shape[0], :patch.shape[1]] = patch
            patch = padded_patch
        
        # Pass the patch through the model
        patch = patch.astype('float32') / 255.0

        # Add the output patch to the list
        output_patches.append((i, j, patch))

# Create a blank output image
output_image = np.zeros(img.shape, dtype=np.float32)

# Combine the output patches into the output image
for i, j, patch in output_patches:
    h, w = patch.shape
    output_image[max(0, i):min(img.shape[0], i+h), max(0, j):min(img.shape[1], j+w)] = patch[:min(h, img.shape[0]-i), :min(w, img.shape[1]-j)]

# Normalize the output image
output_image = (output_image - np.min(output_image)) / (np.max(output_image) - np.min(output_image))

# Compare the original and output images
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(output_image)
plt.title('Output Image')
plt.show()

# Save the output image
cv2.imwrite('output_image.jpg', (output_image * 255).astype(np.uint8))

# Save the low image
cv2.imwrite('low_image.jpg', (img).astype(np.uint8))

print('Done')
