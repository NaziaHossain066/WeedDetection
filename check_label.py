from PIL import Image
import numpy as np
import glob
import os
from tqdm import tqdm  # Import tqdm for the progress bar

def check_unique_colors_in_masks(mask_folder):
    # Get all PNG files in the folder
    mask_paths = glob.glob(os.path.join(mask_folder, "*.png"))

    all_unique_values = set()  # To store all unique pixel values across masks

    # Use tqdm to create a progress bar
    for mask_path in tqdm(mask_paths, desc="Processing masks", unit="mask"):
        # Load the mask image
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale if it's not already

        # Convert mask to a numpy array
        mask_array = np.array(mask)

        # Get unique values in the mask and update the global set
        all_unique_values.update(np.unique(mask_array))

    # Print the unique values found across all masks (sorted)
    print(f"Unique pixel values across all masks: {sorted(all_unique_values)}")

# Replace with the path to your mask folder
mask_folder = r"C:\Users\nazia\Lab\Code\cropandweed-dataset\data\labelIds\CropOrWeed2"
check_unique_colors_in_masks(mask_folder)
