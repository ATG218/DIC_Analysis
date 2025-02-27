import subprocess
import logging
import os
import json
from pathlib import Path
from datetime import datetime

def setup_logging(output_dir):
    """Set up logging configuration."""
    # Create logs directory in the output folder
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger('dice_workflow')
    logger.setLevel(logging.INFO)
    
    # Create a unique log file for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'dice_workflow_{timestamp}.log'
    
    # Create handlers and formatters
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Set formatters for handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def run_command(command, logger):
    """Run a command and log its output."""
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Log output in real-time
        for line in process.stdout:
            logger.info(line.strip())
        
        process.wait()
        return process.returncode == 0
        
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False

def check_existing_files(output_folder):
    """Check if input.xml and params.xml exist in the output folder."""
    input_file = os.path.join(output_folder, 'input.xml')
    params_file = os.path.join(output_folder, 'params.xml')
    return os.path.exists(input_file) and os.path.exists(params_file)

def get_user_input(prompt):
    """Get user input with yes/no validation."""
    while True:
        response = input(prompt + " (y/n): ").lower().strip()
        if response in ['y', 'n']:
            return response == 'y'
        print("Please answer 'yes' or 'no'")

def main():
    try:
        # Load settings
        with open('settings.json', 'r') as f:
            settings = json.load(f)
        
        # Create output directories using relative paths
        os.makedirs(settings['output_folder'], exist_ok=True)
        
        # Setup logging with output directory
        logger = setup_logging(settings['output_folder'])
        logger.info("Starting DICe processing")
        
        # Verify DICe executable
        dice_exe = settings.get('dice_exe_path')
        if not dice_exe:
            raise ValueError("dice_exe_path not found in settings.json")
        if not os.path.exists(dice_exe):
            raise FileNotFoundError(f"DICe executable not found at: {dice_exe}")
        
        # Check for existing files
        files_exist = check_existing_files(settings['output_folder'])
        generate_new_files = True
        
        if files_exist:
            logger.info("Found existing input.xml and params.xml files")
            use_existing = get_user_input("Do you want to use existing configuration files?")
            if use_existing:
                generate_new_files = False
                logger.info("Using existing configuration files")
            else:
                logger.info("Will generate new configuration files")
        
        if generate_new_files:
            # Step 1: Generate DICe configuration files
            logger.info("Generating DICe configuration files...")
            try:
                import generate_dice_files
                generate_dice_files.main()
                logger.info("DICe configuration files generated successfully!")
            except Exception as e:
                logger.error(f"Failed to generate DICe configuration files: {e}")
                raise Exception("Failed to generate DICe configuration files")
        
        # Ask user if they want to proceed with analysis
        proceed = get_user_input("Do you want to run the DICe analysis now?")
        if not proceed:
            logger.info("Analysis skipped by user. You can run post-processing later using dice_post_processing.py")
            return
        
        # Step 2: Run DICe analysis
        logger.info("Running DICe analysis...")
        orig_dir = os.getcwd()
        try:
            os.chdir(settings['output_folder'])
            if not run_command([dice_exe, '-i', 'input.xml'], logger):
                raise Exception("DICe analysis failed")
            logger.info("DICe analysis completed successfully!")
            logger.info("You can now run post-processing using dice_post_processing.py")
        finally:
            os.chdir(orig_dir)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
