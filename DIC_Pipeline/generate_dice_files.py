import os
import glob
import json
import logging
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime

def setup_logging(output_dir=None):
    """Set up logging configuration."""
    logger = logging.getLogger('dice_workflow')
    if not logger.handlers:  # Only add handler if it doesn't already exist
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Add file handler if output directory is provided
        if output_dir:
            log_dir = Path(output_dir) / 'logs'
            log_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f'dice_generate_{timestamp}.log'
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger

def get_image_files(input_folder, reference_path=None):
    """Get sorted list of image files from input folder and reference path."""
    logger = logging.getLogger(__name__)
    
    image_path = Path(input_folder)
    images = sorted([f.name for f in image_path.glob('*.bmp') if f.is_file()])
    if not images:
        raise Exception("No .bmp images found in the specified input folder")
    
    # If reference path is provided and exists, copy it to input folder
    if reference_path and os.path.exists(reference_path):
        try:
            ref_path = Path(reference_path)
            ref_name = ref_path.stem + '_reference' + ref_path.suffix
            ref_dest = image_path / ref_name
            
            # Copy reference image to input folder
            shutil.copy2(reference_path, ref_dest)
            logger.info(f"Successfully copied reference image from {reference_path} to {ref_dest}")
            
            # Add reference image to start of list
            images.insert(0, ref_name)
        except Exception as e:
            logger.error(f"Failed to copy reference image: {str(e)}")
            raise
    return images

def create_input_xml(settings, output_file, images):
    root = ET.Element("ParameterList")
    
    # Get output directory for absolute paths and ensure forward slashes
    output_dir = str(Path(settings['output_folder']).as_posix())
    
    # Add basic parameters with forward slashes
    ET.SubElement(root, "Parameter", name="subset_file", type="string", 
                 value=str(Path(output_dir) / 'subsets.txt').replace('\\', '/'))
    ET.SubElement(root, "Parameter", name="subset_size", type="int", value=str(settings['subset_size']))
    ET.SubElement(root, "Parameter", name="step_size", type="int", value=str(settings['step_size']))
    
    # Create and use results subdirectory
    results_dir = str(Path(output_dir) / 'results').replace('\\', '/') + "/"
    ET.SubElement(root, "Parameter", name="output_folder", type="string", value=results_dir)
    
    # Image folder and params file with forward slashes
    ET.SubElement(root, "Parameter", name="image_folder", type="string", 
                 value=str(Path(settings['input_folder']).as_posix()) + "/")
    ET.SubElement(root, "Parameter", name="correlation_parameters_file", type="string", 
                 value=str(Path(output_dir) / 'params.xml').replace('\\', '/'))
    
    # Use absolute paths for images
    ET.SubElement(root, "Parameter", name="reference_image", type="string", value=images[0])
    
    # Add deformed images list
    if len(images) > 1:
        deformed_list = ET.SubElement(root, "ParameterList", name="deformed_images")
        # Remove duplicates while maintaining order
        unique_images = list(dict.fromkeys(images))
        for img in unique_images:
            ET.SubElement(deformed_list, "Parameter", name=img, type="bool", value="true")
    
    # Write the XML file with proper indentation
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_file, encoding='utf-8', xml_declaration=True)

def create_params_xml(settings, output_file):
    root = ET.Element("ParameterList")
    
    # Add correlation parameters
    analysis_settings = settings['analysis_settings']
    
    # Add initialization method
    ET.SubElement(root, "Parameter", name="initialization_method", type="string", 
                 value=analysis_settings.get('initialization_method', 'USE_NEIGHBOR_VALUES'))
    
    # Add optimization method if specified
    if 'optimization_method' in analysis_settings:
        ET.SubElement(root, "Parameter", name="optimization_method", type="string", 
                     value=analysis_settings['optimization_method'])
    
    ET.SubElement(root, "Parameter", name="sssig_threshold", type="double", 
                 value=str(analysis_settings['sssig_threshold']))
    ET.SubElement(root, "Parameter", name="enable_translation", type="bool", 
                 value=str(analysis_settings['enable_translation']).lower())
    ET.SubElement(root, "Parameter", name="enable_rotation", type="bool", 
                 value=str(analysis_settings['enable_rotation']).lower())
    ET.SubElement(root, "Parameter", name="enable_normal_strain", type="bool", 
                 value=str(analysis_settings['enable_normal_strain']).lower())
    ET.SubElement(root, "Parameter", name="enable_shear_strain", type="bool", 
                 value=str(analysis_settings['enable_shear_strain']).lower())
    ET.SubElement(root, "Parameter", name="output_delimiter", type="string", value=",")
    
    # Add strain window settings
    strain_list = ET.SubElement(root, "ParameterList", name="post_process_vsg_strain")
    ET.SubElement(strain_list, "Parameter", name="strain_window_size_in_pixels", type="int", 
                 value=str(analysis_settings['strain_window_size']))
    
    # Add output specifications
    output_list = ET.SubElement(root, "ParameterList", name="output_spec")
    output_spec = settings.get('output_spec', {})
    for key, value in output_spec.items():
        ET.SubElement(output_list, "Parameter", name=key, type="bool", value=str(value).lower())
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_file, encoding='utf-8', xml_declaration=True)

def create_subsets_txt(settings, output_file):
    roi = settings['region_of_interest']
    with open(output_file, 'w') as f:
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

def main(settings=None):
    try:
        # Load settings if not provided
        if settings is None:
            with open('settings.json', 'r') as f:
                settings = json.load(f)
        
        # Create output directory if it doesn't exist
        output_dir = Path(settings['output_folder'])
        output_dir.mkdir(exist_ok=True)
        
        # Setup logging with output directory
        logger = setup_logging(output_dir)
        logger.info("Starting DICe file generation")
        
        # Get list of image files
        images = get_image_files(settings['input_folder'], settings.get('reference_path'))
        logger.info(f"Found {len(images)} image files")
        
        # Create input.xml
        input_file = output_dir / 'input.xml'
        create_input_xml(settings, input_file, images)
        logger.info(f"Created input.xml at {input_file}")
        
        # Create params.xml
        params_file = output_dir / 'params.xml'
        create_params_xml(settings, params_file)
        logger.info(f"Created params.xml at {params_file}")
        
        # Create subsets.txt
        subsets_file = output_dir / 'subsets.txt'
        create_subsets_txt(settings, subsets_file)
        logger.info(f"Created subsets.txt at {subsets_file}")
        
        # Create results directory
        results_dir = output_dir / 'results'
        results_dir.mkdir(exist_ok=True)
        logger.info(f"Created results directory at {results_dir}")
        
        logger.info("DICe file generation completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to generate DICe files: {e}")
        raise

if __name__ == "__main__":
    main()
