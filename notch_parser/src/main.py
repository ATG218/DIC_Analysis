import logging
import os
from image_processor import ImageProcessor

def read_settings(file_path):
    settings = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=')
            settings[key] = value
    return settings

def setup_logging(log_file):
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f'Log directory created at {log_dir}')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    settings = read_settings('settings.txt')
    
    input_directory = settings.get('input_directory')
    output_directory = settings.get('output_directory')
    log_file = settings.get('log_directory')
    reference_directory = settings.get('reference_directory')

    setup_logging(log_file)
    
    # Log the settings received
    logging.info(f'Settings received: {settings}')
    print(f'Settings received: {settings}')
    
    # Initialize the ImageProcessor
    processor = ImageProcessor(input_directory, output_directory, log_file, reference_directory)
    
    # Start processing images
    processor.process_images()

if __name__ == "__main__":
    main()