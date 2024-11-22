from PIL import Image
import numpy as np
import glob
import os
from tqdm import tqdm



# Assuming you are loading the mask from file
def check_mask_values(mask_path):
    # Open the mask image using Pillow
    mask = Image.open(mask_path)

    # Convert the mask to a numpy array to easily view pixel values
    mask_array = np.array(mask)

    # Print the actual pixel values of the mask
    print("Mask pixel values:")
    print(mask_array)

    # Optionally, display the shape and some stats about the image
    print(f"Shape of mask array: {mask_array.shape}")
    print(f"Unique pixel values: {np.unique(mask_array)}")  # This will show unique values in the mask

# Example usage:
mask_path = r"C:\Users\nazia\Lab\Code\cropandweed-dataset\data\labelIds\CropOrWeed2\ave-0042-0015.png"  # Replace with the actual path # Replace with the actual path
check_mask_values(mask_path)