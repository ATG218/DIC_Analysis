import logging
import os
import json
from pathlib import Path
import plot_results
import strain_analysis
from datetime import datetime
import pandas as pd
import glob

def setup_logging(output_dir):
    """Set up logging configuration."""
    # Create logs directory in the output folder
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger('dice_post_processing')
    logger.setLevel(logging.INFO)
    
    # Create a unique log file for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'dice_post_processing_{timestamp}.log'
    
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

def find_first_solution_file(results_dir):
    """Find the first DICe solution file in the results directory."""
    # Use glob to find all solution files and sort them
    solution_pattern = os.path.join(results_dir, "results", "DICe_solution_*.txt")
    solution_files = sorted(glob.glob(solution_pattern))
    print(solution_files[0])
    return solution_files[0] if solution_files else None

def get_user_input(prompt):
    """Get user input with yes/no validation."""
    while True:
        response = input(prompt + " (y/n): ").lower().strip()
        if response in ['y', 'n']:
            return response == 'y'
        print("Please answer 'y' or 'n'")

def get_folder_name(input_folder):
    """Extract the last component of the input folder path."""
    folder_name = os.path.basename(os.path.normpath(input_folder))
    if not folder_name:  # Handle case where path ends with separator
        folder_name = os.path.basename(os.path.dirname(input_folder))
    return folder_name.replace(" ", "_")  # Replace spaces with underscores for filenames

def save_strain_data(df, folder_name, subset_id, output_dir, logger):
    """Save strain data to a CSV file."""
    # Create CSV filename with subset ID only
    csv_file = output_dir / f'subset_{subset_id}_principal_data.csv'
    
    # Prepare data for CSV
    data = {
        'Frame': df['Time'],
        'Exx': df['Exx'],  # Normal strain in X direction
        'Eyy': df['Eyy'],  # Normal strain in Y direction
        'Exy': df['Exy'],  # Shear strain
        'e1': df['e1'],    # Maximum principal strain
        'e2': df['e2']     # Minimum principal strain
    }
    
    # Save to CSV
    pd.DataFrame(data).to_csv(csv_file, index=False)
    logger.info(f"Strain data saved to CSV: {csv_file}")
    logger.info(f"Data includes {len(df)} frames for subset ID {subset_id}")
    logger.info("Columns: Frame number, Normal strains (Exx, Eyy), Shear strain (Exy), Principal strains (e1, e2)")

def main():
    try:
        # Load settings
        with open('settings.json', 'r') as f:
            settings = json.load(f)
        
        # Get output directory (parent of DICe results)
        output_dir = Path(settings['output_folder'])
        
        # Setup logging with output directory
        logger = setup_logging(output_dir)
        logger.info("Starting DICe post-processing")
        
        # Get folder name from input folder path
        folder_name = get_folder_name(settings['input_folder'])
        settings_name = Path('settings.json').name
        logger.info(f"Processing data for folder: {folder_name} using settings {settings_name}")
        
        # Find first solution file
        first_solution = find_first_solution_file(settings['output_folder'])
        if not first_solution:
            raise FileNotFoundError("No initial solution file found! (checked both DICe_solution_0.txt and DICe_solution_00.txt)")
        
        # Step 1: Generate visualization
        logger.info("Generating tracking points visualization...")
        try:
            # Update visualization filenames to include folder name and output directory
            settings['visualization_settings']['output_prefix'] = str(output_dir / f"{folder_name}_tracking_points")
            plot_results.plot_tracking_points(first_solution, settings)
            logger.info(f"Visualization generated successfully for {folder_name} using settings {settings_name}")
        except Exception as e:
            logger.error(f"Failed to generate visualization: {e}")
            raise
        
        # Ask user if they want to proceed with strain analysis
        proceed = get_user_input("Do you want to perform principal strain analysis?")
        if not proceed:
            logger.info("Principal strain analysis skipped by user")
            return
        
        # Step 2: Run principal strain analysis
        logger.info("Starting principal strain analysis...")
        try:
            # Get the strain data
            subset_id = strain_analysis.get_valid_id(first_solution)
            if subset_id is None:
                logger.info("Strain analysis cancelled by user")
                return
                
            strain_data = strain_analysis.process_all_solutions(settings['output_folder'], subset_id)
            if strain_data is None:
                raise Exception("Failed to process strain data")
            
            # Update settings to include subset ID only in output filenames
            settings['visualization_settings']['output_prefix'] = str(output_dir / f"subset_{subset_id}_principal_strains")
            
            # Generate plots
            strain_analysis.plot_principal_strains(strain_data, subset_id, settings)
            logger.info(f"Principal strain analysis plots generated successfully for subset {subset_id} using settings {settings_name}")
            
            # Save strain data to CSV
            save_strain_data(strain_data, folder_name, subset_id, output_dir, logger)
            
        except Exception as e:
            logger.error(f"Failed to run principal strain analysis: {e}")
            raise
        
        logger.info(f"DICe post-processing completed successfully using settings {settings_name}")
        
    except Exception as e:
        logger.error(f"Post-processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
