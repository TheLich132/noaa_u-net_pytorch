from model import UNet
import os
import torch
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import NOAAAPTDataset
from patchify import patchify
from torchmetrics.functional import structural_similarity_index_measure as ssim
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch_optimizer import RAdam  # Import RAdam optimizer
import torch.nn.functional as F
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio as psnr

class SSIML1Loss(torch.nn.Module):
    def __init__(self, alpha=0.8):
        """
        Custom loss function combining SSIM and L1 loss.
        
        Parameters:
        - alpha: Weight for SSIM loss (0 to 1); (1 - alpha) will be the weight for L1 loss.
        """
        super(SSIML1Loss, self).__init__()
        self.alpha = alpha
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, output, target):
        # Calculate SSIM (higher SSIM is better, so we take 1 - SSIM to treat it as a loss)
        ssim_loss = 1 - ssim(output, target, data_range=1.0)  # data_range=1 if image is normalized [0, 1]
        
        # Calculate L1 Loss
        l1_loss = self.l1_loss(output, target)
        
        # Combine the losses
        combined_loss = self.alpha * ssim_loss + (1 - self.alpha) * l1_loss
        return combined_loss
    
class PerceptualTVLoss(nn.Module):
    def __init__(self, tv_weight=1e-5, perceptual_weight=1.0):
        """
        Combined Perceptual and Total Variation (TV) Loss.
        
        Parameters:
        - tv_weight: Weight for TV Loss
        - perceptual_weight: Weight for Perceptual Loss
        """
        super(PerceptualTVLoss, self).__init__()
        self.tv_weight = tv_weight
        self.perceptual_weight = perceptual_weight

        # Load a pre-trained VGG model for feature extraction
        vgg = models.vgg16(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg)[:16]).eval()  # Use only the first few layers
        self.feature_extractor.to(device)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze VGG parameters

    def tv_loss(self, image):
        """
        Compute the total variation loss (TV loss) for an image.
        """
        tv_h = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
        return tv_h + tv_w

    def perceptual_loss(self, output, target):
        """
        Compute the perceptual loss between output and target.
        """
        if output.shape[1] == 1:  # Check if it's grayscale
            output = output.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        output = output.to(device)
        target = target.to(device)

        output_features = self.feature_extractor(output)
        target_features = self.feature_extractor(target)
        return F.mse_loss(output_features, target_features)

    def forward(self, output, target):
        # Calculate perceptual loss
        perceptual_loss = self.perceptual_loss(output, target)
        
        # Calculate TV loss
        tv_loss = self.tv_loss(output)

        # Combine losses
        total_loss = self.perceptual_weight * perceptual_loss + self.tv_weight * tv_loss
        return total_loss



if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

torch.cuda.empty_cache()

high_quality_path = r'/var/home/jakub/Obrazy/apt_training_set/high_quality'
low_quality_path = r'/var/home/jakub/Obrazy/apt_training_set/low_quality'

# Load NOAA APT images from path
def load_images_from_folder(folder, size=(256, 256)):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, size)
            images.append(img)
    return images

high_quality = load_images_from_folder(high_quality_path)
low_quality = load_images_from_folder(low_quality_path)

high_quality_img_patches = []

for img in high_quality:
    patches_img = patchify(img, (256, 256), step=256)
    
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i,j,:,:]
            single_patch_img = (single_patch_img.astype('float32')) / 255.0

            high_quality_img_patches.append(single_patch_img)

high_quality_images = np.array(high_quality_img_patches)

low_quality_img_patches = []

for img in low_quality:
    patches_img = patchify(img, (256, 256), step=256)
    
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i,j,:,:]
            single_patch_img = (single_patch_img.astype('float32')) / 255.0

            low_quality_img_patches.append(single_patch_img)

low_quality_images = np.array(low_quality_img_patches)

low_quality_train, low_quality_val, high_quality_train, high_quality_val = train_test_split(low_quality_images, high_quality_images, test_size=0.2, random_state=42)

train_dataset = NOAAAPTDataset(low_quality_train, high_quality_train)
val_dataset = NOAAAPTDataset(low_quality_val, high_quality_val)

train_loader = DataLoader(train_dataset, batch_size=22, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=22, shuffle=False, num_workers=2, pin_memory=True)

model = UNet(1, in_channels=1, start_filts=75, depth=4)
# model = nn.DataParallel(model)
model = model.to(device)

# criterion = torch.nn.MSELoss()
# criterion = torch.nn.L1Loss()
criterion = SSIML1Loss(alpha=0.85)
# criterion = PerceptualTVLoss(tv_weight=1e-5, perceptual_weight=1.0)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
# optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4)

def calculate_psnr(output, target):
    # Convert tensors to numpy arrays for PSNR calculation
    output_np = output.cpu().detach().numpy()
    target_np = target.cpu().detach().numpy()
    return psnr(target_np, output_np, data_range=target_np.max() - target_np.min())

for epoch in range(50):
    for batch in train_loader:
        low_quality_images, high_quality_images = batch

        low_quality_images = low_quality_images.to(device)
        high_quality_images = high_quality_images.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(low_quality_images)
        loss = criterion(outputs, high_quality_images)

        # Backward pass
        loss.backward()

        # Update the model parameters
        optimizer.step()

    # Validate the model
    model.eval()
    with torch.no_grad():
        val_loss = 0
        total_psnr = 0  # To accumulate PSNR values
        num_batches = 0  # To count batches for averaging

        for batch in val_loader:
            low_quality_images, high_quality_images = batch

            low_quality_images = low_quality_images.to(device)
            high_quality_images = high_quality_images.to(device)

            outputs = model(low_quality_images)
            loss = criterion(outputs, high_quality_images)
            val_loss += loss.item()

            # Calculate PSNR for each batch
            psnr_value = calculate_psnr(outputs, high_quality_images)
            total_psnr += psnr_value
            num_batches += 1

        avg_psnr = total_psnr / num_batches if num_batches > 0 else 0
        print(f'Epoch {epoch + 1}, Val Loss: {val_loss / len(val_loader)}, Avg PSNR: {avg_psnr:.4f}')

    model.train()

# save pytorch model
torch.save(model.state_dict(), 'unet_model.pt')