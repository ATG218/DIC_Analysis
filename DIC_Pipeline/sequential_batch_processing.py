import json
import logging
import os
from pathlib import Path
import subprocess
from typing import List, Dict, Tuple, Any
import sys
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

def setup_logging():
    """Setup logging configuration"""
    try:
        # Create logs directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / 'sequential_processing.log'
    except Exception as e:
        # Fallback to default logs directory
        log_file = Path('logs') / 'sequential_processing.log'
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

def extract_cycle_info(filename: str) -> Tuple[int, int]:
    """Extract cycle number and index from a filename.
    Expected format: Notch_XX_cycle_YYYYY_ZZZZZ.bmp
    Returns a tuple of (cycle_number, index_number)
    """
    pattern = r'cycle_(\d+)_(\d+)'
    match = re.search(pattern, filename)
    if match:
        cycle_num = int(match.group(1))
        index_num = int(match.group(2))
        return cycle_num, index_num
    else:
        raise ValueError(f"Could not extract cycle information from filename: {filename}")

def group_files_by_cycle(input_folder: str) -> Dict[int, List[str]]:
    """Group image files by cycle number
    Returns a dictionary where keys are cycle numbers and values are lists of file paths sorted by index
    """
    cycle_groups = defaultdict(list)
    
    # Get all BMP files in the input folder
    files = [f for f in os.listdir(input_folder) if f.lower().endswith('.bmp')]
    
    # Group files by cycle number
    for file in files:
        try:
            cycle_num, index_num = extract_cycle_info(file)
            full_path = os.path.join(input_folder, file)
            cycle_groups[cycle_num].append((full_path, index_num))
        except ValueError as e:
            # Skip files that don't match the expected pattern
            continue
    
    # Sort files within each cycle by index number
    for cycle in cycle_groups:
        cycle_groups[cycle].sort(key=lambda x: x[1])
    
    # Convert list of tuples to list of filenames (keeping the sorted order)
    result = {cycle: [path for path, _ in files] for cycle, files in cycle_groups.items()}
    
    return result

def generate_parameter_combinations(notch_settings: Dict) -> List[Dict]:
    """Generate all combinations of subset sizes, step sizes, and strain window multiples"""
    # Get parameter lists (or convert single values to lists)
    subset_sizes = notch_settings.get('subset_sizes', [notch_settings.get('subset_size', 21)])
    step_sizes = notch_settings.get('step_sizes', [notch_settings.get('step_size', 5)])
    strain_multiples = notch_settings.get('strain_window_multiples', [3])
    
    # Ensure all parameters are lists
    if not isinstance(subset_sizes, list):
        subset_sizes = [subset_sizes]
    if not isinstance(step_sizes, list):
        step_sizes = [step_sizes]
    if not isinstance(strain_multiples, list):
        strain_multiples = [strain_multiples]
    
    combinations = []
    
    # Generate all combinations of parameters
    for subset in subset_sizes:
        for step in step_sizes:
            for multiple in strain_multiples:
                # Calculate actual strain window size
                strain_window_size = step * multiple
                
                # Create a new settings dictionary with these parameters
                settings_copy = notch_settings.copy()
                settings_copy['subset_size'] = subset
                settings_copy['step_size'] = step
                
                # Deep copy the analysis_settings to avoid modifying the original
                if 'analysis_settings' in settings_copy:
                    settings_copy['analysis_settings'] = settings_copy['analysis_settings'].copy()
                    settings_copy['analysis_settings']['strain_window_size'] = strain_window_size
                
                combinations.append(settings_copy)
    
    return combinations

def generate_json_config(settings: Dict, cycle_files: List[str], cycle_num: int,
                      output_dir: str, logger: logging.Logger) -> str:
    """Generate JSON configuration for a specific cycle
    
    Args:
        settings: Dictionary containing configuration settings
        cycle_files: List of image files for the cycle
        cycle_num: Cycle number
        output_dir: Output directory for JSON files
        logger: Logger instance
        
    Returns:
        Path to the generated JSON file
    """
    # Extract required parameters from settings
    dice_exe_path = settings['dice_exe_path']
    subset_size = settings['subset_size']
    step_size = settings['step_size']
    strain_window_size = settings['analysis_settings']['strain_window_size']
    roi = settings['region_of_interest']
    
    # Get reference image (first image in cycle)
    reference_path = cycle_files[0]
    
    # Create unique output folder for this cycle and parameter combination
    cycle_output_folder = os.path.join(
        output_dir, 
        f"cycle_{cycle_num:05d}_subset{subset_size:02d}_step{step_size:02d}_VSG{strain_window_size:03d}"
    )
    
    # Create directory if it doesn't exist
    os.makedirs(cycle_output_folder, exist_ok=True)
    # Create results subdirectory (required by DICe)
    results_dir = os.path.join(cycle_output_folder, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a cycle-specific settings object
    analysis_settings = settings['analysis_settings'].copy()
    analysis_settings['strain_window_size'] = strain_window_size  # Ensure this specific VSG is used
    
    # Base structure of the JSON
    config = {
        "dice_exe_path": dice_exe_path,
        "reference_path": reference_path,
        "input_folder": os.path.dirname(reference_path),  # Use original folder for input
        "output_folder": cycle_output_folder,
        "subset_size": subset_size,
        "step_size": step_size,
        "analysis_settings": analysis_settings,
        "output_spec": settings['output_spec'],
        "region_of_interest": roi,
        "visualization_settings": settings.get('visualization_settings', {}),
        "cycle_files": cycle_files  # Add list of files to process for this cycle
    }
    
    # Create unique filename for this cycle's config with parameter information
    json_filename = os.path.join(
        output_dir, 
        f"cycle_{cycle_num:05d}_subset{subset_size:02d}_step{step_size:02d}_VSG{strain_window_size:03d}_config.json"
    )
    
    # Write the JSON file
    with open(json_filename, 'w') as json_file:
        json.dump(config, json_file, indent=4)
    
    logger.info(f"Generated configuration for cycle {cycle_num} with parameters: subset={subset_size}, step={step_size}, VSG={strain_window_size}")
    
    return json_filename

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

def run_dice_analysis(config_file: str, logger: logging.Logger) -> bool:
    """Run DICe executable for a cycle analysis"""
    try:
        start_time = time.time()
        logger.info(f"Starting DICe analysis using config file '{config_file}'")
        
        # Load config
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        dice_exe = config['dice_exe_path']
        output_folder = config['output_folder']
        
        # First, generate all the needed DICe input files - using our cycle-specific generator
        generate_cycle_dice_files(config, logger)
        
        # Now the input.xml file should be in the output folder
        input_xml_path = os.path.join(output_folder, 'input.xml')
        
        if not os.path.exists(dice_exe):
            raise FileNotFoundError(f"DICe executable not found at {dice_exe}")
        
        if not os.path.exists(input_xml_path):
            raise FileNotFoundError(f"DICe input file not found at {input_xml_path}")
        
        # Run DICe with real-time output logging
        logger.info(f"Running DICe on {input_xml_path}")
        success = run_command([dice_exe, '-i', str(input_xml_path)], logger)
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"DICe analysis completed for {config_file} in {duration:.2f} seconds")
        
        if not success:
            raise Exception("DICe analysis failed. Check the log for details.")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to run DICe for {config_file}: {str(e)}")
        return False

def generate_cycle_dice_files(settings: Dict, logger: logging.Logger) -> None:
    """Generate DICe input files for a specific cycle"""
    try:
        # Create output directory structure
        output_dir = Path(settings['output_folder'])
        output_dir.mkdir(exist_ok=True)
        results_dir = output_dir / 'results'
        results_dir.mkdir(exist_ok=True)
        
        # Extract settings
        subset_size = settings['subset_size']
        step_size = settings['step_size']
        strain_window_size = settings['analysis_settings']['strain_window_size']
        roi = settings['region_of_interest']
        output_spec = settings['output_spec']
        cycle_files = settings.get('cycle_files', [])
        
        # Ensure we have cycle files
        if not cycle_files or len(cycle_files) < 2:
            raise ValueError("Not enough cycle files provided")
        
        # Get reference image (first image in cycle)
        reference_image = os.path.basename(cycle_files[0])
        
        # Create input.xml file
        input_xml_path = output_dir / 'input.xml'
        
        # Get only the basenames of all files
        image_basenames = [os.path.basename(f) for f in cycle_files]
        
        # Create XML content for input.xml
        root = ET.Element("ParameterList")
        
        # Add basic parameters
        ET.SubElement(root, "Parameter", name="subset_file", type="string", 
                     value=str(output_dir / 'subsets.txt').replace('\\', '/'))
        ET.SubElement(root, "Parameter", name="subset_size", type="int", value=str(subset_size))
        ET.SubElement(root, "Parameter", name="step_size", type="int", value=str(step_size))
        
        # Create and use results subdirectory
        results_path = str(output_dir / 'results').replace('\\', '/') + "/"
        ET.SubElement(root, "Parameter", name="output_folder", type="string", value=results_path)
        
        # Image folder with forward slashes
        input_folder = str(Path(settings['input_folder']).as_posix()) + "/"
        ET.SubElement(root, "Parameter", name="image_folder", type="string", value=input_folder)
        ET.SubElement(root, "Parameter", name="correlation_parameters_file", type="string", 
                     value=str(output_dir / 'params.xml').replace('\\', '/'))
        
        # Reference image
        ET.SubElement(root, "Parameter", name="reference_image", type="string", value=reference_image)
        
        # Add deformed images list - only include images from this specific cycle
        if len(image_basenames) > 1:
            deformed_list = ET.SubElement(root, "ParameterList", name="deformed_images")
            # Skip the first image (reference) and include only unique filenames
            unique_images = list(dict.fromkeys(image_basenames[1:]))
            for img in unique_images:
                ET.SubElement(deformed_list, "Parameter", name=img, type="bool", value="true")
        
        # Write the XML file with proper indentation
        if hasattr(ET, 'indent'):
            ET.indent(root)
        tree = ET.ElementTree(root)
        tree.write(input_xml_path, encoding='utf-8', xml_declaration=True)
        
        # Create params.xml file
        params_xml_path = output_dir / 'params.xml'
        
        # Create XML content for params.xml
        params_root = ET.Element("ParameterList")
        
        # Add correlation parameters from analysis_settings
        analysis_settings = settings['analysis_settings']
        
        # Add initialization method
        ET.SubElement(params_root, "Parameter", name="initialization_method", type="string", 
                     value=analysis_settings.get('initialization_method', 'USE_NEIGHBOR_VALUES'))
        
        # Add optimization method if specified
        if 'optimization_method' in analysis_settings:
            ET.SubElement(params_root, "Parameter", name="optimization_method", type="string", 
                         value=analysis_settings['optimization_method'])
        
        ET.SubElement(params_root, "Parameter", name="sssig_threshold", type="double", 
                     value=str(analysis_settings['sssig_threshold']))
        ET.SubElement(params_root, "Parameter", name="enable_translation", type="bool", 
                     value=str(analysis_settings['enable_translation']).lower())
        ET.SubElement(params_root, "Parameter", name="enable_rotation", type="bool", 
                     value=str(analysis_settings['enable_rotation']).lower())
        ET.SubElement(params_root, "Parameter", name="enable_normal_strain", type="bool", 
                     value=str(analysis_settings['enable_normal_strain']).lower())
        ET.SubElement(params_root, "Parameter", name="enable_shear_strain", type="bool", 
                     value=str(analysis_settings['enable_shear_strain']).lower())
        ET.SubElement(params_root, "Parameter", name="output_delimiter", type="string", value=",")
        
        # Add strain window settings
        strain_list = ET.SubElement(params_root, "ParameterList", name="post_process_vsg_strain")
        ET.SubElement(strain_list, "Parameter", name="strain_window_size_in_pixels", type="int", 
                     value=str(strain_window_size))
        
        # Add output specifications
        output_list = ET.SubElement(params_root, "ParameterList", name="output_spec")
        for key, value in output_spec.items():
            ET.SubElement(output_list, "Parameter", name=key, type="bool", value=str(value).lower())
        
        # Write the params XML file
        if hasattr(ET, 'indent'):
            ET.indent(params_root)
        params_tree = ET.ElementTree(params_root)
        params_tree.write(params_xml_path, encoding='utf-8', xml_declaration=True)
        
        # Create subsets.txt file
        subsets_path = output_dir / 'subsets.txt'
        
        with open(subsets_path, 'w') as f:
            f.write("begin region_of_interest\n")
            f.write("  begin boundary\n")
            
            if roi['type'].lower() == 'rectangle':
                f.write("    begin rectangle\n")
                f.write(f"      center {roi['center'][0]} {roi['center'][1]}\n")
                f.write(f"      width {roi['width']}\n")
                f.write(f"      height {roi['height']}\n")
                f.write("    end rectangle\n")
            elif roi['type'].lower() == 'polygon':
                f.write("    begin polygon\n")
                f.write("      begin vertices\n")
                for vertex in roi['vertices']:
                    f.write(f"        {vertex[0]} {vertex[1]}\n")
                f.write("      end vertices\n")
                f.write("    end polygon\n")
            
            f.write("  end boundary\n")
            
            if 'excluded' in roi:
                f.write("  begin excluded\n")
                for excluded in roi['excluded']:
                    if excluded['type'].lower() == 'rectangle':
                        f.write("    begin rectangle\n")
                        f.write(f"      center {excluded['center'][0]} {excluded['center'][1]}\n")
                        f.write(f"      width {excluded['width']}\n")
                        f.write(f"      height {excluded['height']}\n")
                        f.write("    end rectangle\n")
                    elif excluded['type'].lower() == 'polygon':
                        f.write("    begin polygon\n")
                        f.write("      begin vertices\n")
                        for vertex in excluded['vertices']:
                            f.write(f"        {vertex[0]} {vertex[1]}\n")
                        f.write("      end vertices\n")
                        f.write("    end polygon\n")
                f.write("  end excluded\n")
            
            f.write("end region_of_interest\n")
        
        logger.info(f"Successfully generated cycle-specific DICe files in {output_dir}")
        
    except Exception as e:
        logger.error(f"Failed to generate cycle-specific DICe files: {str(e)}")
        raise

def process_notch(notch_settings: Dict, logger: logging.Logger) -> bool:
    """Process a single notch with all its cycles and parameter combinations"""
    try:
        start_time = time.time()
        input_folder = notch_settings['input_folder']
        output_base = notch_settings['output_folder']
        
        logger.info(f"Processing notch: {os.path.basename(input_folder)}")
        
        # Group files by cycle
        cycle_groups = group_files_by_cycle(input_folder)
        logger.info(f"Found {len(cycle_groups)} cycles in {input_folder}")
        
        if len(cycle_groups) == 0:
            logger.warning(f"No valid image cycles found in {input_folder}")
            return False
        
        # Generate all parameter combinations
        parameter_combinations = generate_parameter_combinations(notch_settings)
        logger.info(f"Generated {len(parameter_combinations)} parameter combinations")
        
        # Generate JSON config for each cycle and parameter combination
        config_files = []
        for cycle_num, cycle_files in cycle_groups.items():
            if len(cycle_files) < 2:  # Need at least 2 images for an analysis
                logger.warning(f"Cycle {cycle_num} has only {len(cycle_files)} images, skipping")
                continue
                
            # For each cycle, process all parameter combinations
            for settings in parameter_combinations:
                config_file = generate_json_config(
                    settings, 
                    cycle_files, 
                    cycle_num, 
                    output_base, 
                    logger
                )
                config_files.append(config_file)
        
        # Run DICe analyses in parallel
        success = True
        use_parallel = notch_settings.get('parallel_processing', True)
        
        if use_parallel:
            logger.info(f"Running DICe analyses in parallel for {len(config_files)} configurations...")
            num_workers = os.cpu_count()
            logger.info(f"Using {num_workers} workers")
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(run_dice_analysis, config_file, logger) 
                          for config_file in config_files]
                           
                for future in concurrent.futures.as_completed(futures):
                    try:
                        if not future.result():
                            success = False
                    except Exception as e:
                        logger.error(f"Analysis failed: {str(e)}")
                        success = False
        else:
            logger.info(f"Running DICe analyses sequentially for {len(config_files)} configurations...")
            for config_file in config_files:
                if not run_dice_analysis(config_file, logger):
                    success = False
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Processed notch {os.path.basename(input_folder)} in {duration:.2f} seconds")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to process notch: {str(e)}")
        return False

def main():
    logger = setup_logging()
    start_time = time.time()
    logger.info("Starting sequential batch processing")
    
    try:
        # Load the sequential batch settings
        settings_file = 'sequential_batch_settings.json'
        
        if not os.path.exists(settings_file):
            logger.error(f"Settings file '{settings_file}' not found. Please create it first.")
            print(f"Settings file '{settings_file}' not found. Please create it first.")
            return False
        
        with open(settings_file, 'r') as f:
            batch_settings = json.load(f)
        
        # Load notch settings
        notch_settings = batch_settings.get('notch_settings', [])
        if not notch_settings:
            logger.error("No notch settings found in the configuration")
            print("No notch settings found in the configuration")
            return False
        
        # Generate configuration files for each notch without running analysis
        all_config_files = []
        for i, settings in enumerate(notch_settings):
            logger.info(f"Processing notch {i+1} of {len(notch_settings)}")
            
            # Extract required variables
            input_folder = settings['input_folder']
            output_base = settings['output_folder']
            
            # Group files by cycle
            cycle_groups = group_files_by_cycle(input_folder)
            logger.info(f"Found {len(cycle_groups)} cycles in {input_folder}")
            
            if len(cycle_groups) == 0:
                logger.warning(f"No valid image cycles found in {input_folder}")
                continue
            
            # Generate all parameter combinations
            parameter_combinations = generate_parameter_combinations(settings)
            logger.info(f"Generated {len(parameter_combinations)} parameter combinations")
            
            # Generate JSON config for each cycle and parameter combination
            notch_config_files = []
            for cycle_num, cycle_files in cycle_groups.items():
                if len(cycle_files) < 2:  # Need at least 2 images for an analysis
                    logger.warning(f"Cycle {cycle_num} has only {len(cycle_files)} images, skipping")
                    continue
                    
                # For each cycle, process all parameter combinations
                for param_settings in parameter_combinations:
                    config_file = generate_json_config(
                        param_settings, 
                        cycle_files, 
                        cycle_num, 
                        output_base, 
                        logger
                    )
                    notch_config_files.append(config_file)
            
            all_config_files.extend(notch_config_files)
            logger.info(f"Generated {len(notch_config_files)} configuration files for notch {i+1}")
        
        # Pause and ask user if they want to proceed with analysis
        total_configs = len(all_config_files)
        print(f"\nGenerated {total_configs} configuration files for analysis.")
        response = input("\nWould you like to proceed with DICe analysis? (y/n): ")
        
        if response.lower() != 'y':
            logger.info("DICe analysis cancelled by user")
            print("DICe analysis cancelled")
            return True
        
        # Run DICe analyses in parallel for all notches
        success = True
        for i, settings in enumerate(notch_settings):
            # Filter config files for this notch
            notch_base = os.path.basename(settings['input_folder'])
            notch_output = settings['output_folder']
            notch_configs = [f for f in all_config_files if notch_output in f]
            
            if not notch_configs:
                logger.info(f"No configurations to process for notch {notch_base}")
                continue
                
            # Run DICe analyses in parallel
            use_parallel = settings.get('parallel_processing', True)
            
            if use_parallel:
                logger.info(f"Running DICe analyses in parallel for {len(notch_configs)} configurations...")
                num_workers = os.cpu_count()
                logger.info(f"Using {num_workers} workers")
                
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(run_dice_analysis, config_file, logger) 
                              for config_file in notch_configs]
                               
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            if not future.result():
                                success = False
                        except Exception as e:
                            logger.error(f"Analysis failed: {str(e)}")
                            success = False
            else:
                logger.info(f"Running DICe analyses sequentially for {len(notch_configs)} configurations...")
                for config_file in notch_configs:
                    if not run_dice_analysis(config_file, logger):
                        success = False
            
            logger.info(f"Completed analysis for notch {notch_base}")
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Total processing time: {duration:.2f} seconds")
        
        if success:
            logger.info("Sequential batch processing completed successfully")
            print("Sequential batch processing completed successfully")
        else:
            logger.error("Sequential batch processing completed with errors")
            print("Sequential batch processing completed with errors. Check the log for details.")
        
    except Exception as e:
        logger.error(f"Sequential batch processing failed: {str(e)}")
        print(f"Sequential batch processing failed: {str(e)}")
        return False
    
    return success

def create_example_settings():
    """Create an example settings file if none exists"""
    settings_file = 'sequential_batch_settings.json'
    
    if os.path.exists(settings_file):
        print(f"Settings file '{settings_file}' already exists. Not overwriting.")
        return
    
    # Example settings for multiple notches with parameter combinations
    example_settings = {
        "notch_settings": [
            {
                "dice_exe_path": "C:/Program Files (x86)/Digital Image Correlation Engine/dice.exe",
                "input_folder": "C:/Users/alext/OneDrive/Documents/UROP25/data/OneDrive_2025-05-09/PH_17-4_sorted/Notch_01",
                "output_folder": "C:/Users/alext/OneDrive/Documents/UROP25/data/OneDrive_2025-05-09/PH_17-4_sorted/Notch_01/DICe_sequential",
                "subset_sizes": [17, 19, 21, 23, 25],             # List of subset sizes
                "step_sizes": [2, 3, 5],                         # List of step sizes
                "strain_window_multiples": [2, 3, 5, 7],         # List of multiples (actual VSG = step_size * multiple)
                "analysis_settings": {
                    "sssig_threshold": 500,
                    "enable_translation": True,
                    "enable_rotation": True,
                    "enable_normal_strain": True,
                    "enable_shear_strain": True,
                    "optimization_method": "GRADIENT_BASED",
                    "initialization_method": "USE_FEATURE_MATCHING",
                    "output_delimiter": ",",
                    "strain_window_size": 15  # This will be overridden by combination calculations
                },
                "output_spec": {
                    "COORDINATE_X": True,
                    "COORDINATE_Y": True,
                    "VSG_STRAIN_XX": True,
                    "VSG_STRAIN_YY": True,
                    "VSG_STRAIN_XY": True,
                    "GAMMA": True,
                    "SIGMA": True,
                    "STATUS_FLAG": True
                },
                "region_of_interest": {
                    "type": "polygon",
                    "vertices": [
                        [511, 200],
                        [510, 214],
                        [510, 224],
                        [513, 231],
                        [518, 239],
                        [524, 244],
                        [535, 249],
                        [543, 251],
                        [557, 251],
                        [564, 247],
                        [572, 241],
                        [578, 233],
                        [583, 223],
                        [583, 210],
                        [583, 199],
                        [600, 199],
                        [600, 275],
                        [495, 275],
                        [495, 200]
                    ]
                },
                "visualization_settings": {
                    "plot_type": "scatter",
                    "data_columns": {
                        "x": "COORDINATE_X",
                        "y": "COORDINATE_Y",
                        "id": "SUBSET_ID"
                    },
                    "display_options": {
                        "show_ids": True,
                        "point_size": 8,
                        "point_color": "blue",
                        "point_opacity": 0.6,
                        "label_size": 8,
                        "show_grid": True
                    },
                    "save_formats": ["html", "png"]
                },
                "parallel_processing": True
            },
            # Example for Notch_02 with different ROI and parameters
            {
                "dice_exe_path": "C:/Program Files (x86)/Digital Image Correlation Engine/dice.exe",
                "input_folder": "C:/Users/alext/OneDrive/Documents/UROP25/data/OneDrive_2025-05-09/PH_17-4_sorted/Notch_02",
                "output_folder": "C:/Users/alext/OneDrive/Documents/UROP25/data/OneDrive_2025-05-09/PH_17-4_sorted/Notch_02/DICe_sequential",
                "subset_sizes": [17, 19, 21],                     # Different subset sizes for Notch_02
                "step_sizes": [3, 5],                             # Different step sizes for Notch_02
                "strain_window_multiples": [3, 5],                # Different strain window multiples for Notch_02
                "analysis_settings": {
                    "sssig_threshold": 500,
                    "enable_translation": True,
                    "enable_rotation": True,
                    "enable_normal_strain": True,
                    "enable_shear_strain": True,
                    "optimization_method": "GRADIENT_BASED",
                    "initialization_method": "USE_FEATURE_MATCHING",
                    "output_delimiter": ",",
                    "strain_window_size": 15  # This will be overridden by combination calculations
                },
                "output_spec": {
                    "COORDINATE_X": True,
                    "COORDINATE_Y": True,
                    "VSG_STRAIN_XX": True,
                    "VSG_STRAIN_YY": True,
                    "VSG_STRAIN_XY": True,
                    "GAMMA": True,
                    "SIGMA": True,
                    "STATUS_FLAG": True
                },
                "region_of_interest": {
                    "type": "polygon",
                    "vertices": [
                        [350, 250],
                        [650, 250],
                        [650, 750],
                        [350, 750]
                    ]
                },
                "visualization_settings": {
                    "plot_type": "scatter",
                    "data_columns": {
                        "x": "COORDINATE_X",
                        "y": "COORDINATE_Y",
                        "id": "SUBSET_ID"
                    },
                    "display_options": {
                        "show_ids": True,
                        "point_size": 8,
                        "point_color": "blue",
                        "point_opacity": 0.6,
                        "label_size": 8,
                        "show_grid": True
                    },
                    "save_formats": ["html", "png"]
                },
                "parallel_processing": True
            }
        ]
    }
    
    with open(settings_file, 'w') as f:
        json.dump(example_settings, f, indent=4)
    
    print(f"Example settings file created at '{settings_file}'")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--create-example':
        create_example_settings()
    else:
        main()
