#%% 
import sys
import os
CTI_SCRIPTS = os.getenv('CTI_SCRIPTS')
print(f'CTI_SCRIPTS={CTI_SCRIPTS}')
sys.path.append(f'{CTI_SCRIPTS}')
sys.path.append(f'{CTI_SCRIPTS}/cti')
sys.path.append(f'{CTI_SCRIPTS}/pythonModules')

import cti_image
from pyCascade.cti.lesCreateMovie import (
    process_image, create_cbar, cbar_padding,
    validate_parameters, get_nticks
)

from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import os


#%% Method 3: Reusable function leveraging lesCreateMovie components
def create_image_with_colorbar(
    image, output_path, cmaplist, data_min, data_max, 
    cbar_title="Data", cbar_orient="vertical",
    cbar_width_frac=0.05, cbar_height_frac=0.8,
    nticks=None, fontsize=14,
    background_color=[255, 255, 255] # white
):
    """Create and save an image with a properly formatted colorbar
    
    Args:
        image: processed image
        output_path: Path to save the image
        cmaplist: List of colormaps to use
        data_min: List of minimum values for colormaps
        data_max: List of maximum values for colormaps
        cbar_title: Title for the colorbar
        cbar_orient: Orientation of colorbar ("horizontal" or "vertical")
        cbar_width_frac: Width of colorbar as fraction of image width
        cbar_height_frac: Height of colorbar as fraction of image height
        nticks: Number of ticks on colorbar
        fontsize: Font size for colorbar text
        background_color: RGB color for background (default: bahama blue)
    
    Returns:
        Combined image with colorbar
    """

    # Calculate dimensions
    dpi = 80
    image_width = image.shape[1]
    image_height = image.shape[0]
    cbar_width = cbar_width_frac * image_width / dpi
    cbar_height = cbar_height_frac * image_height / dpi
    
    # Create temporary colorbar file
    temp_cbar_path = "temp_colorbar.png"

    # infer number of ticks
    if nticks is None:
       nticks = get_nticks(data_min[0], data_max[0])
    
    # Create colorbar - passing parameters explicitly
    create_cbar(data_min[0], data_max[0], cmaplist[0], nticks, cbar_title, 
               temp_cbar_path, cbar_width, cbar_height, cbar_orient,
               background_color, fontsize)
    
    # Load and process colorbar - passing parameters explicitly
    cbar_img = [cv.imread(temp_cbar_path)]
    cb_resized = cbar_padding(None, image.shape[0], image.shape[1], 
                             cbar_img, 1, cbar_orient, background_color)
    
    # Combine image and colorbar
    if cbar_orient == 'horizontal':
        combined_image = np.vstack((image, cb_resized[0]))
    else:  # vertical
        combined_image = np.hstack((image, cb_resized[0]))
    
    # Save the combined image
    if output_path is not None:
        cv.imwrite(output_path, combined_image)
    print(f"Saved image with colorbar to {output_path}")
    
    # Clean up temporary file
    if os.path.exists(temp_cbar_path):
        os.remove(temp_cbar_path)
    
    return combined_image

# %%

def main():
    # Define path to image and parameters
    image_path = "/oak/stanford/groups/gorle/nbachand/Cascade/city_block_cfd/CHARLES/config3/temp_images/R16_mean_u_y1p5.00240000.png"
    output_dir = "/oak/stanford/groups/gorle/nbachand/Cascade/city_block_cfd/CHARLES/config3/temp_images/python_images"
    image_name = os.path.basename(image_path)
    # os.makedirs(output_dir, exist_ok=True)

    varlist=['planar']
    cmaplist=['plasma']
    data_min=[0]
    data_max=[2]

     # Process the image
    image = process_image(image_path, varlist, cmaplist, data_min, data_max)

    # Example usage of the reusable function
    custom_output = os.path.join(output_dir, image_name)
    result = create_image_with_colorbar(
        image, custom_output, cmaplist, data_min, data_max, 
        cbar_title="Velocity (m/s)",
        cbar_orient="vertical"
    )

    # Display the result
    plt.figure(figsize=(12, 10))
    plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Result from Reusable Function")
    plt.show()

if __name__ == "__main__":
    main()
