import torch
import cv2
import numpy as np
from patchify import patchify, unpatchify
from model import UNet
import matplotlib.pyplot as plt
import torch.nn as nn

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = UNet(1, in_channels=1, start_filts=75, depth=4)
# model = nn.DataParallel(model)
model.load_state_dict(torch.load('unet_model.pt', weights_only=True))
model.to(device)
model.eval()

# Load the low-quality image
low_image = cv2.imread('/var/home/jakub/Obrazy/apt_training_set/low_quality/low_noaa-15-201111181355-norm.jpg', cv2.IMREAD_GRAYSCALE)
actual_image = cv2.imread('/var/home/jakub/Obrazy/apt_training_set/high_quality/noaa-18-201703120714-norm.jpg', cv2.IMREAD_GRAYSCALE)

# Define the patch size and overlap
patch_size = 256
overlap = 128

# Calculate the step size
step_size = patch_size - overlap

# Create a list to hold the output patches
output_patches = []

# Create a grid of patch positions
for i in range(0, low_image.shape[0], step_size):
    for j in range(0, low_image.shape[1], step_size):
        # Extract the patch from the image
        patch = low_image[max(0, i):min(low_image.shape[0], i+patch_size), max(0, j):min(low_image.shape[1], j+patch_size)]

        # Pad the patch to the full patch size if necessary
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            padded_patch = np.zeros((patch_size, patch_size), dtype=np.uint8)
            padded_patch[:patch.shape[0], :patch.shape[1]] = patch
            patch = padded_patch
        
        # Pass the patch through the model
        patch = patch.astype('float32') / 255.0
        patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
        output_patch = model(patch)
        output_patch = output_patch.squeeze(0).squeeze(0).cpu().detach().numpy()

        # Add the output patch to the list
        output_patches.append((i, j, output_patch))

# Create a blank output image
output_image = np.zeros(low_image.shape, dtype=np.float32)

# Combine the output patches into the output image
for i, j, patch in output_patches:
    h, w = patch.shape
    output_image[max(0, i):min(low_image.shape[0], i+h), max(0, j):min(low_image.shape[1], j+w)] = patch[:min(h, low_image.shape[0]-i), :min(w, low_image.shape[1]-j)]

# Normalize the output image
output_image = (output_image - np.min(output_image)) / (np.max(output_image) - np.min(output_image))

# Compare the output image with the actual image
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(low_image, cmap='gray')
plt.title('Low Quality Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(output_image, cmap='gray')
plt.title('Output Image')
plt.axis('off')
plt.show()

# Save the output image
cv2.imwrite('output_image.jpg', (output_image * 255).astype(np.uint8))

# Save the low image
cv2.imwrite('low_image.jpg', (low_image).astype(np.uint8))