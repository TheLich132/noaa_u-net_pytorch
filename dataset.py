import torch
from torch.utils.data import Dataset

class NOAAAPTDataset(Dataset):
    def __init__(self, low_quality_images, high_quality_images):
        self.low_quality_images = low_quality_images
        self.high_quality_images = high_quality_images

    def __len__(self):
        return len(self.low_quality_images)

    def __getitem__(self, index):
        low_quality_image = self.low_quality_images[index]
        high_quality_image = self.high_quality_images[index]

        # Convert images to tensors
        low_quality_tensor = torch.from_numpy(low_quality_image).unsqueeze(0).float()
        high_quality_tensor = torch.from_numpy(high_quality_image).unsqueeze(0).float()

        return low_quality_tensor, high_quality_tensor