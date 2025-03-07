import os
import shutil
from pathlib import Path
import re

def natural_sort_key(s):
    """Sort strings containing numbers in natural order."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

def organize_images_for_notch_parser(input_dir, output_dir, images_per_cycle=8):
    """
    Organize images into folders of 8 for notch parser compatibility.
    
    Args:
        input_dir (str): Directory containing all images
        output_dir (str): Directory where organized folders will be created
        images_per_cycle (int): Number of images per cycle (default 8)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files and sort them
    image_files = []
    for ext in ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        image_files.extend([f for f in Path(input_dir).glob(f'*{ext}')])
        image_files.extend([f for f in Path(input_dir).glob(f'*{ext.upper()}')])
    
    # Remove duplicates and sort images naturally
    image_files = list(set(image_files))
    image_files.sort(key=natural_sort_key)
    
    # Calculate number of cycles
    total_images = len(image_files)
    if total_images == 0:
        print("No images found in input directory!")
        return
    
    if total_images % images_per_cycle != 0:
        print(f"Warning: Total number of images ({total_images}) is not divisible by {images_per_cycle}")
    
    # Create cycle folders and copy images
    for i, image_path in enumerate(image_files):
        cycle_num = i // images_per_cycle
        cycle_folder = os.path.join(output_dir, f't_{cycle_num:04d}')
        os.makedirs(cycle_folder, exist_ok=True)
        
        # Copy image to new location
        dest_path = os.path.join(cycle_folder, image_path.name)
        shutil.copy2(image_path, dest_path)
        print(f'Copied {image_path.name} to t_{cycle_num:04d}')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize images into folders for notch parser')
    parser.add_argument('input_dir', help='Directory containing input images')
    parser.add_argument('output_dir', help='Directory where organized folders will be created')
    parser.add_argument('--images-per-cycle', type=int, default=8,
                        help='Number of images per cycle (default: 8)')
    
    args = parser.parse_args()
    
    organize_images_for_notch_parser(args.input_dir, args.output_dir, args.images_per_cycle)
