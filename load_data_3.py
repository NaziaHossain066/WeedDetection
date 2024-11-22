import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets.cropandweed_dataset_2 import CustomDataset
import numpy as np

# Set up paths to your dataset root directory
dataset_root_dir = 'C:\\Users\\nazia\Lab\Code\cropandweed-dataset\data'  # Replace this with the actual path

# Instantiate the CustomDataset
dataset = CustomDataset(root_dir=dataset_root_dir)

# Check the length of the dataset
print(f"Total images in dataset: {len(dataset)}")

# Create a DataLoader to iterate over the dataset
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Iterate over the first 10 samples and print out details
for idx, (image, mask) in enumerate(dataloader):
    if image is None or mask is None:
        continue  # Skip any missing mask cases

    # Print image and mask shapes
    print(f"Sample {idx + 1}:")
    print(f"  Image shape: {image.shape}")
    print(f"  Mask shape: {mask.shape}")

    # Convert tensors to NumPy arrays for visualization
    image_np = image[0].permute(1, 2, 0).numpy().astype(np.uint8)  # Convert back to HxWxC
    mask_np = mask[0].numpy()

    # Plot image and mask side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image_np)
    ax1.set_title("Image")
    ax2.imshow(mask_np, cmap="gray")
    ax2.set_title("Mask")
    plt.show()

    # Stop after displaying the first 10 samples
    if idx >= 9:
        break
