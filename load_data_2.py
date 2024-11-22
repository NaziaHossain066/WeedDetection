import os

import torch
from torch.utils.data import Dataset
from PIL import Image

from torchvision import transforms
from torch.utils.data import DataLoader

class CropOrWeedDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")  # Assuming masks are grayscale

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


#-------------------------------------------------------------------------------------------------------------------------------------------

# Paths to image and mask folders
image_folder = 'C:\\Users\\nazia\Lab\Code\cropandweed-dataset\data\images'
mask_folder = 'C:\\Users\\nazia\Lab\Code\cropandweed-dataset\data\labelIds\CropOrWeed2'

# Get set of image and mask filenames (without extensions)
image_files = {f.rsplit('.', 1)[0] for f in os.listdir(image_folder) if f.endswith('.jpg')}
mask_files = {f.rsplit('.', 1)[0] for f in os.listdir(mask_folder) if f.endswith('.png')}

# Find intersection of base filenames
matched_files = image_files.intersection(mask_files)
print(f"Number of matched files: {len(matched_files)}")

# Construct full paths for matched image and mask files
image_paths = [os.path.join(image_folder, f + '.jpg') for f in matched_files]
mask_paths = [os.path.join(mask_folder, f + '.png') for f in matched_files]

print(len(image_paths), len(image_paths))

#-----------------------------------------------------------------------------------------------------------------------------------------

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize if needed
    transforms.ToTensor(),          # Convert to tensor
])

# Create the dataset and dataloader
dataset = CropOrWeedDataset(images=image_paths, masks=mask_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Test the DataLoader
for images, masks in dataloader:
    print(f"Image batch shape: {images.size()}")
    print(f"Mask batch shape: {masks.size()}")
    break


