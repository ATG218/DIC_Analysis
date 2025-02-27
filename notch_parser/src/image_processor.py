import os
import logging
import time
import shutil
import cv2
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class ImageProcessor:
    def __init__(self, input_directory, output_directory, log_file, reference_directory):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.log_file = log_file
        self.reference_directory = reference_directory
        self.setup_logging()
        self.reference_images = []
        self.sorted_notches = []
        self.timestamps = []
        self.notch_order_map = {}  # Maps original index to width-based index

    def setup_logging(self):
        log_dir = os.path.dirname(self.log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            logging.info(f'Log directory created at {log_dir}')
        logging.basicConfig(filename=self.log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def create_output_directories(self):
        for i in range(1, 9):
            dir_path = os.path.join(self.output_directory, f'Notch_{i:02d}')
            os.makedirs(dir_path, exist_ok=True)
        os.makedirs(self.reference_directory, exist_ok=True)
        logging.info(f'Output directories created at {self.output_directory}')

    def load_images(self, folder):
        image_files = [f for f in os.listdir(folder) if f.endswith('.bmp')]
        if len(image_files) != 8:
            logging.warning(f'Expected 8 images in folder {folder}, but found {len(image_files)}.')
        return image_files

    def calculate_notch_width(self, binary):
        # Convert to uint8 if not already
        binary = binary.astype(np.uint8) * 255
        
        height, width = binary.shape
        
        # Start looking at 14% down from the top
        start_y = height // 4
        search_height = height // 4  # Look through the middle 25%
        
        # Create ROI in the search area
        roi = binary[start_y:start_y + search_height, :]
        
        # Create mask of bright pixels (value > 240)
        bright_mask = (roi > 240).astype(np.uint8) * 255
        
        # Create debug image
        debug_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        
        min_width = 50  # Minimum expected notch width
        max_width = 300  # Maximum expected notch width
        
        # For each row in the search area
        for y in range(search_height):
            # Get the bright pixels in this row
            row_values = roi[y, :]
            bright_pixels = np.where(row_values > 240)[0]
            
            if len(bright_pixels) > 0:
                # Find continuous regions of bright pixels
                gaps = np.diff(bright_pixels)
                gap_threshold = 5  # Allow small gaps
                
                # Split into continuous regions
                split_indices = np.where(gaps > gap_threshold)[0] + 1
                regions = np.split(bright_pixels, split_indices)
                
                # Find the widest continuous region
                max_width_found = 0
                best_left = 0
                best_right = 0
                
                for region in regions:
                    if len(region) > 0:
                        left_edge = region[0]
                        right_edge = region[-1]
                        region_width = right_edge - left_edge
                        
                        if region_width > max_width_found:
                            max_width_found = region_width
                            best_left = left_edge
                            best_right = right_edge
                
                # Verify the width is reasonable
                if min_width < max_width_found < max_width:
                    # Draw the detected width on debug image
                    cv2.line(debug_img, (best_left, y), (best_right, y), (0, 255, 0), 2)
                    cv2.circle(debug_img, (best_left, y), 3, (255, 0, 0), -1)
                    cv2.circle(debug_img, (best_right, y), 3, (255, 0, 0), -1)
                    
                    # Save debug images
                    cv2.imwrite('debug_notch_width.png', debug_img)
                    cv2.imwrite('debug_edges.png', bright_mask)
                    
                    actual_y = start_y + y
                    logging.info(f'Found notch at row {actual_y} with width {max_width_found}')
                    return max_width_found
        
        logging.warning('No valid notch edges found')
        return 0

    def move_reference_images(self, reference_folder):
        image_files = self.load_images(reference_folder)
        
        # Store original images with their widths
        notch_widths = []
        for i, image_file in enumerate(image_files):
            source_path = os.path.join(reference_folder, image_file)
            image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
            width = self.calculate_notch_width(image)
            notch_widths.append((i, width))
            logging.info(f'Original notch {i} has width: {width}')
        
        # Sort by width and create mapping (smallest to largest)
        sorted_notches = sorted(notch_widths, key=lambda x: x[1])  # Sort by width
        self.notch_order_map = {}
        
        # Create mapping: original_index -> new_index (1-based)
        for new_idx, (orig_idx, width) in enumerate(sorted_notches):
            self.notch_order_map[orig_idx] = new_idx + 1
        
        # Now copy images to reference directory with correct ordering
        for i, image_file in enumerate(image_files):
            source_path = os.path.join(reference_folder, image_file)
            new_index = self.notch_order_map[i]
            dest_path = os.path.join(self.reference_directory, f'Notch_{new_index:02d}.bmp')
            shutil.copy2(source_path, dest_path)
            image = cv2.imread(dest_path, cv2.IMREAD_GRAYSCALE)
            self.reference_images.append(image)
        
        logging.info(f'Reference images moved to {self.reference_directory} with width-based ordering')
        logging.info(f'Final mapping: {self.notch_order_map}')

    def process_images(self):
        start_time = time.time()
        self.create_output_directories()

        subfolders = [f.path for f in os.scandir(self.input_directory) if f.is_dir()]
        if not subfolders:
            logging.error('No subfolders found in the input directory.')
            return

        reference_folder = subfolders[0]
        # Extract the index from the reference folder name
        ref_folder_index = self.extract_index_number(reference_folder)
        self.move_reference_images(reference_folder)

        # Copy reference images to their respective sorted folders with original index
        for orig_idx, notch_idx in self.notch_order_map.items():
            ref_image_path = os.path.join(self.reference_directory, f'Notch_{notch_idx:02d}.bmp')
            if os.path.exists(ref_image_path):
                dest_folder = os.path.join(self.output_directory, f'Notch_{notch_idx:02d}')
                dest_path = os.path.join(dest_folder, f'Notch_{notch_idx:02d}_cycle_0000_{ref_folder_index}.bmp')
                shutil.copy2(ref_image_path, dest_path)
                logging.info(f'Copied reference image to {dest_path}')

        total_subfolders = len(subfolders[1:])
        print(f"\nProcessing {total_subfolders} subfolders...")
        
        with ThreadPoolExecutor() as executor:
            futures = []
            with tqdm(total=total_subfolders, desc="Processing Subfolders", unit="folder") as pbar:
                for subfolder in subfolders[1:]:
                    future = executor.submit(self.process_subfolder, subfolder, pbar)
                    futures.append(future)
                
                for future in futures:
                    future.result()  # Ensure all threads have completed

        # Add index to filenames after sorting
        # print("\nAdding indices to filenames...")
        # self.add_index_to_filenames()

        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f'Total processing time: {total_time:.2f} seconds')
        print(f'\nTotal processing time: {total_time:.2f} seconds')

        # Write timestamps to a single text file
        print("Writing timestamps...")
        self.write_timestamps()

        # Delete the reference directory
        print("Cleaning up reference directory...")
        self.delete_reference_directory()
        
        print("\nProcessing complete!")

    def process_subfolder(self, subfolder, pbar=None):
        try:
            image_files = self.load_images(subfolder)
            cycle_number = self.extract_cycle_number(subfolder)
            index_number = self.extract_index_number(subfolder)
            timestamp = self.extract_timestamp(subfolder)
            self.timestamps.append(timestamp)

            # Process the first image to determine the starting notch
            first_image_path = os.path.join(subfolder, image_files[0])
            first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
            best_match_index, best_match_score = self.find_best_match(first_image)
            logging.info(f'Processing subfolder {subfolder} (Cycle {cycle_number})')

            # Process all images with progress tracking
            for i, image_file in enumerate(image_files):
                source_image_path = os.path.join(subfolder, image_file)
                try:
                    image = cv2.imread(source_image_path, cv2.IMREAD_GRAYSCALE)
                    original_index = (best_match_index + i) % 8
                    notch_index = self.notch_order_map[original_index]  # Map to width-based index
                    destination_image_path = os.path.join(
                        self.output_directory, 
                        f'Notch_{notch_index:02d}',
                        f'Notch_{notch_index:02d}_cycle_{cycle_number}_{index_number}.bmp'
                    )
                    cv2.imwrite(destination_image_path, image)
                except Exception as e:
                    logging.error(f'Error processing image {source_image_path}: {e}')
        except Exception as e:
            logging.error(f'Error processing subfolder {subfolder}: {e}')
        finally:
            if pbar:
                pbar.update(1)

    def add_index_to_filenames(self):
        for notch_index in range(1, 9):
            notch_dir = os.path.join(self.output_directory, f'Notch_{notch_index:02d}')
            image_files = sorted([f for f in os.listdir(notch_dir) if f.endswith('.bmp')])
            for index, image_file in enumerate(image_files):
                old_path = os.path.join(notch_dir, image_file)
                parts = image_file.split('_')
                cycle_number = parts[3]
                new_filename = f'Notch_{notch_index:02d}_cycle_{cycle_number}_{index:05d}.bmp'
                new_path = os.path.join(notch_dir, new_filename)
                if old_path != new_path:
                    if os.path.exists(new_path):
                        logging.warning(f'File {new_path} already exists. Skipping rename.')
                    else:
                        os.rename(old_path, new_path)

    def extract_timestamp(self, folder):
        txt_files = [f for f in os.listdir(folder) if f.endswith('.txt')]
        if txt_files:
            with open(os.path.join(folder, txt_files[0]), 'r') as file:
                timestamp = file.read().strip()
            return timestamp.replace(':', '-')
        else:
            logging.warning(f'No timestamp file found in folder: {folder}')
            return "unknown"

    def extract_cycle_number(self, folder):
        folder_name = os.path.basename(folder)
        cycle_number = folder_name.split('_')[-1]
        return cycle_number

    def extract_index_number(self, folder):
        folder_name = os.path.basename(folder)
        index_number = folder_name.split('_')[1]
        return index_number    

    def find_best_match(self, image):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(image, None)

        best_match_index = -1
        best_match_score = float('inf')  # Lower score is better for distance

        for i, ref_image in enumerate(self.reference_images):
            kp2, des2 = orb.detectAndCompute(ref_image, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            score = sum([match.distance for match in matches]) / len(matches) if matches else float('inf')
            if score < best_match_score:
                best_match_score = score
                best_match_index = i

        return best_match_index, best_match_score

    def write_timestamps(self):
        timestamps_file = os.path.join(self.output_directory, 'timestamps.txt')
        with open(timestamps_file, 'w') as file:
            for timestamp in self.timestamps:
                file.write(f'{timestamp}\n')
        logging.info(f'Timestamp file created at {timestamps_file}')

    def delete_reference_directory(self):
        try:
            shutil.rmtree(self.reference_directory)
            logging.info(f'Reference directory {self.reference_directory} deleted successfully.')
        except Exception as e:
            logging.error(f'Error deleting reference directory {self.reference_directory}: {e}')