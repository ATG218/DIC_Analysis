import json
import logging
import os
from pathlib import Path
import subprocess
from typing import List, Dict
import sys
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import time

def setup_logging():
    """Setup logging configuration"""
    try:
        # Get output directory from first settings file in batch_settings
        with open('batch_settings.json', 'r') as f:
            batch_settings = json.load(f)
            settings_files = batch_settings.get('settings_files', [])
            
            if settings_files:
                with open(settings_files[0], 'r') as sf:
                    settings = json.load(sf)
                output_path = Path(settings['output_folder'])
                # Create parent directory first
                output_path.parent.mkdir(parents=True, exist_ok=True)
                log_dir = output_path.parent / 'logs'
                log_dir.mkdir(exist_ok=True)
                log_file = log_dir / 'batch_processing.log'
            else:
                # Fallback to default logs directory
                log_file = Path('logs') / 'batch_processing.log'
                log_file.parent.mkdir(exist_ok=True)
    except Exception as e:
        # Fallback to default logs directory
        log_file = Path('logs') / 'batch_processing.log'
        log_file.parent.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def load_settings(settings_file: str) -> Dict:
    """Load and validate a settings file"""
    try:
        with open(settings_file, 'r') as f:
            settings = json.load(f)
            
        required_fields = ['dice_exe_path', 'input_folder', 'output_folder']
        for field in required_fields:
            if field not in settings:
                raise ValueError(f"Missing required field '{field}' in {settings_file}")
                
        # Create output directory structure if it doesn't exist
        output_path = Path(settings['output_folder'])
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Create parent dir (e.g., 20250129_304)
        output_path.mkdir(parents=True, exist_ok=True)  # Create output dir (e.g., notch1)
            
        return settings
    except Exception as e:
        raise Exception(f"Error loading {settings_file}: {str(e)}")

def check_folder_conflicts(settings_list: List[Dict]) -> List[str]:
    """Check for duplicate input or output folders"""
    input_folders = {}
    output_folders = {}
    conflicts = []
    
    for i, settings in enumerate(settings_list):
        input_path = Path(settings['input_folder'])
        output_path = Path(settings['output_folder'])
        
        input_name = input_path.name
        if input_name in input_folders:
            conflicts.append(f"Duplicate input folder '{input_name}' found in multiple settings")
        input_folders[input_name] = i
        
        if output_path in output_folders:
            conflicts.append(f"Duplicate output folder '{output_path}' found in multiple settings")
        output_folders[output_path] = i
    
    return conflicts

def generate_dice_files(settings: Dict, logger: logging.Logger) -> None:
    """Generate DICe input files"""
    try:
        # Create output directory structure
        output_dir = Path(settings['output_folder'])
        output_dir.mkdir(exist_ok=True)
        results_dir = output_dir / 'results'
        results_dir.mkdir(exist_ok=True)
        
        # Generate files in the specified output directory
        from generate_dice_files import main as generate_main
        generate_main(settings)
        logger.info(f"Successfully generated DICe files in {output_dir}")
    except Exception as e:
        logger.error(f"Failed to generate DICe files: {str(e)}")
        raise

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

def run_dice_analysis(settings: Dict, logger: logging.Logger, settings_file: str = None) -> bool:
    """Run DICe executable for a single analysis"""
    try:
        start_time = time.time()
        settings_info = f"settings file '{settings_file}'" if settings_file else f"folder '{Path(settings['input_folder']).name}'"
        logger.info(f"Starting DICe analysis using {settings_info}")
        
        dice_exe = settings['dice_exe_path']
        output_dir = Path(settings['output_folder'])
        input_file = output_dir / 'input.xml'
        
        if not os.path.exists(dice_exe):
            raise FileNotFoundError(f"DICe executable not found at {dice_exe}")
            
        # Run DICe with real-time output logging
        success = run_command([dice_exe, '-i', str(input_file)], logger)
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"DICe analysis completed for {settings_info} in {duration:.2f} seconds")
        
        if not success:
            raise Exception("DICe analysis failed. Check the log for details.")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to run DICe for {settings_info}: {str(e)}")
        return False

def main():
    logger = setup_logging()
    start_time = time.time()
    logger.info("Starting batch processing")
    
    try:
        # Load batch settings
        with open('batch_settings.json', 'r') as f:
            batch_settings = json.load(f)
        
        # Check if subset IDs match settings files
        settings_count = len(batch_settings['settings_files'])
        subset_count = len(batch_settings.get('subset_ids', []))
        
        can_post_process = True
        if settings_count != subset_count:
            logger.warning(f"Number of settings files ({settings_count}) does not match number of subset IDs ({subset_count})")
            logger.warning("Post-processing will not be available without matching subset IDs")
            response = input("\nWould you like to proceed with DICe processing only? (y/n): ")
            if response.lower() != 'y':
                logger.info("Batch processing cancelled by user")
                return
            can_post_process = False
        
        # Load all settings files
        settings_list = []
        for settings_file in batch_settings['settings_files']:
            logger.info(f"Loading settings from {settings_file}")
            settings = load_settings(settings_file)
            settings_list.append(settings)
        
        # Check for conflicts
        conflicts = check_folder_conflicts(settings_list)
        if conflicts:
            logger.warning("The following conflicts were detected:")
            for conflict in conflicts:
                logger.warning(conflict)
                
            # List all input folders
            logger.info("\nInput folders detected:")
            for settings in settings_list:
                input_path = Path(settings['input_folder'])
                logger.info(f"- {input_path.name}")
            
            response = input("\nWould you like to continue? (y/n): ")
            if response.lower() != 'y':
                logger.info("Batch processing cancelled by user")
                return
        
        # Generate all DICe input files first
        logger.info("\nGenerating DICe input files for all analyses...")
        for settings in settings_list:
            input_path = Path(settings['input_folder'])
            logger.info(f"Generating files for {input_path.name}")
            generate_dice_files(settings, logger)
        
        # Ask user to continue
        response = input("\nAll DICe input files have been generated. Would you like to proceed with DICe analysis? (y/n): ")
        if response.lower() != 'y':
            logger.info("DICe analysis cancelled by user")
            return
        
        # Ask about parallel processing
        use_parallel = batch_settings.get('parallel_processing', False)
        
        # Ask about post-processing only if subset IDs are available
        do_post_processing = False
        if can_post_process:
            do_post_processing = input("\nWould you like to perform post-processing after DICe analysis? (y/n): ").lower() == 'y'
        
        # Run DICe analyses
        success = True
        if use_parallel:
            logger.info("Running DICe analyses in parallel...")
            num_workers = os.cpu_count()
            logger.info(f"Running batch processing in parallel mode with {num_workers} workers")
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(run_dice_analysis, settings, logger, settings_file) 
                          for settings, settings_file in zip(settings_list, batch_settings['settings_files'])]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        if not future.result():
                            success = False
                    except Exception as e:
                        logger.error(f"Analysis failed: {str(e)}")
                        success = False
        else:
            logger.info("Running DICe analyses sequentially...")
            for settings, settings_file in zip(settings_list, batch_settings['settings_files']):
                if not run_dice_analysis(settings, logger, settings_file):
                    success = False
        
        # Run post-processing if requested and possible
        if do_post_processing and can_post_process:
            logger.info("Running post-processing...")
            from batch_post_processing import batch_post_process, create_combined_strain_plot
            all_strains = []
            for i, settings_file in enumerate(batch_settings['settings_files']):
                try:
                    strain_data = batch_post_process(settings_file, batch_settings['subset_ids'][i], logger)
                    all_strains.append(strain_data)
                except Exception as e:
                    logger.error(f"Error processing {settings_file}: {str(e)}")
                    continue
            
            # Create combined plot if we have any data
            if all_strains:
                try:
                    output_dir = Path('strain_analysis')
                    output_dir.mkdir(exist_ok=True)
                    create_combined_strain_plot(all_strains, output_dir, logger)
                except Exception as e:
                    logger.error(f"Error creating combined plot: {str(e)}")
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Total batch processing time: {duration:.2f} seconds")
        logger.info(f"Average time per analysis: {duration/len(settings_list):.2f} seconds")
        if success:
            logger.info("Batch processing completed successfully")
        else:
            logger.error("Batch processing failed")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
