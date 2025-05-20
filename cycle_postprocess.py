#!/usr/bin/env python3
"""
cycle_postprocess.py - Digital Image Correlation (DIC) Analysis for Multiple Notches

This script performs strain analysis on DIC data across multiple notch samples.
It allows selection of specific cycle, subset, step, and VSG parameters for analysis.
"""

import os
import sys
import glob
import json
import logging
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector, Button
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

# Constants
DEFAULT_DATA_DIR = r"C:\Users\alext\OneDrive\Documents\UROP25\data\OneDrive_2025-05-09\PH_17-4_sorted_2"
DEFAULT_SETTINGS_FILE = "cycle_settings.json"
STATUS_FLAG_OK = 4
GAMMA_THRESHOLD = 0.03  # Default threshold for gamma validation
SIGMA_THRESHOLD = 0.05  # Default threshold for sigma validation

def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"cycle_postprocess_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def load_settings(settings_file=DEFAULT_SETTINGS_FILE):
    """
    Load settings from JSON file or create default settings if file doesn't exist.
    
    Args:
        settings_file (str): Path to settings file.
        
    Returns:
        dict: Settings dictionary.
    """
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            settings = json.load(f)
    else:
        # Default settings
        settings = {
            "data_dir": DEFAULT_DATA_DIR,             # Path to the data directory containing notch folders
            "cycle": "00020",                      # Cycle number (e.g., '00020')
            "subset": "29",                       # Subset size (e.g., '29')
            "step": "03",                         # Step size (e.g., '03')
            "vsg": "009",                         # VSG number (e.g., '009')
            "camera_log_file": "",                # Path to the Camera_Log.txt file (for timestamp data)
            "use_timestamps": False,                 # Whether to use timestamps from camera log file instead of frame numbers
            "gamma_threshold": GAMMA_THRESHOLD,        # Maximum acceptable gamma value
            "sigma_threshold": SIGMA_THRESHOLD,        # Maximum acceptable sigma value
            "output_dir": os.path.join(DEFAULT_DATA_DIR, "analysis_results"),  # Where to save results
            "save_csv": True,                        # Whether to save CSV files of strain data
            "create_plots": True,                     # Whether to create strain plots
            "interactive_plots": True,                # Generate interactive HTML plots
            "static_plots": True,                     # Generate static PNG plots
            "selected_points": {}                     # Dictionary to store selected points for each notch
        }
        
        # Save default settings (JSON doesn't support comments)
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=4)
        
        # Create a separate README file with instructions for the settings
        readme_file = os.path.join(os.path.dirname(settings_file), "cycle_settings_README.txt")
        with open(readme_file, 'w') as f:
            f.write("Cycle Post-Processing Settings Guide\n")
            f.write("=================================\n\n")
            f.write("This file provides instructions for configuring the cycle_settings.json file.\n\n")
            f.write("Settings Parameters:\n")
            f.write("-------------------\n")
            f.write("- data_dir: Path to the main data directory containing notch folders\n")
            f.write("- cycle: Cycle number (e.g., '20' or '00020')\n")
            f.write("- subset: Subset size (e.g., '29')\n")
            f.write("- step: Step size (e.g., '3' or '03')\n")
            f.write("- vsg: VSG number (e.g., '9' or '009')\n")
            f.write("- camera_log_file: Path to the Camera_Log.txt file containing timestamp data\n")
            f.write("- use_timestamps: Set to true to use real timestamps in plots instead of frame numbers\n")
            f.write("- gamma_threshold: Maximum acceptable gamma value for validation\n")
            f.write("- sigma_threshold: Maximum acceptable sigma value for validation\n")
            f.write("- output_dir: Directory where analysis results will be saved\n")
            f.write("- save_csv: Set to true to save strain data as CSV files\n")
            f.write("- create_plots: Set to true to create strain plots\n")
            f.write("- interactive_plots: Set to true to generate interactive HTML plots\n")
            f.write("- static_plots: Set to true to generate static PNG plots\n")
            f.write("- selected_points: DO NOT EDIT manually - populated by the script when selecting points\n")
        
        print(f"Created new settings file at {settings_file}")
        print(f"Created settings guide at {readme_file}")
        print("Please review and edit the settings before running the script again.")
    
    return settings

def save_settings(settings, settings_file=DEFAULT_SETTINGS_FILE):
    """
    Save settings to JSON file.
    
    Args:
        settings (dict): Settings dictionary.
        settings_file (str): Path to settings file.
    """
    # Convert any numpy types to Python native types to avoid JSON serialization issues
    clean_settings = {}
    for key, value in settings.items():
        if key == 'selected_points':
            # Handle the selected_points dictionary separately
            clean_selected_points = {}
            for notch, point_ids in value.items():
                # Handle multiple points per notch (list of points)
                if isinstance(point_ids, list):
                    clean_points = []
                    for pid in point_ids:
                        # Convert numpy int types to Python int
                        if hasattr(pid, 'item') and callable(getattr(pid, 'item')):
                            clean_points.append(pid.item())
                        else:
                            clean_points.append(pid)
                    clean_selected_points[notch] = clean_points
                elif point_ids is not None:
                    # For backward compatibility with single point selection
                    if hasattr(point_ids, 'item') and callable(getattr(point_ids, 'item')):
                        clean_selected_points[notch] = point_ids.item()
                    else:
                        clean_selected_points[notch] = point_ids
                else:
                    clean_selected_points[notch] = None
            clean_settings[key] = clean_selected_points
        else:
            # Handle other settings values
            if hasattr(value, 'item') and callable(getattr(value, 'item')):  # For numpy types
                clean_settings[key] = value.item()
            else:
                clean_settings[key] = value
    
    # Write to file
    with open(settings_file, 'w') as f:
        json.dump(clean_settings, f, indent=4)

def find_notch_folders(data_dir):
    """
    Find all notch folders in the data directory.
    
    Args:
        data_dir (str): Path to data directory.
        
    Returns:
        list: List of paths to notch folders.
    """
    notch_pattern = os.path.join(data_dir, "Notch_*")
    notch_folders = sorted(glob.glob(notch_pattern))
    return notch_folders

def normalize_parameter(value, padding=5):
    """
    Normalize a parameter value by converting to string and adding leading zeros if needed.
    
    Args:
        value (str or int): Parameter value (e.g., '20' or 20 for cycle).
        padding (int): Number of digits to pad to (default: 5 for cycle, others use different padding).
        
    Returns:
        str: Normalized value with leading zeros (e.g., '00020').
    """
    # Convert to string if it's an integer
    value_str = str(value)
    # Strip any leading zeros to handle case where user already provided formatted value
    value_str = value_str.lstrip('0')
    # Add leading zeros only if needed
    if value_str:
        return value_str.zfill(padding)
    else:
        return '0'.zfill(padding)  # Handle case of input '0' or 0

def get_analysis_folder_path(notch_folder, cycle, subset, step, vsg):
    """
    Construct the path to the DICe sequential analysis folder.
    
    Args:
        notch_folder (str): Path to notch folder.
        cycle (str or int): Cycle number (e.g., '20' or 20, will be formatted as '00020').
        subset (str or int): Subset size (e.g., '25').
        step (str or int): Step size (e.g., '3', will be formatted as '03').
        vsg (str or int): VSG number (e.g., '9', will be formatted as '009').
        
    Returns:
        str: Path to analysis folder.
    """
    # Normalize parameters
    cycle_norm = normalize_parameter(cycle, 5)  # e.g., '00020'
    subset_norm = normalize_parameter(subset, 2) if subset else subset  # e.g., '25'
    step_norm = normalize_parameter(step, 2) if step else step  # e.g., '03'
    vsg_norm = normalize_parameter(vsg, 3) if vsg else vsg  # e.g., '009'
    
    folder_name = f"cycle_{cycle_norm}_subset{subset_norm}_step{step_norm}_VSG{vsg_norm}"
    return os.path.join(notch_folder, "DICe_sequential", folder_name)

def get_results_folder(analysis_folder):
    """
    Get the path to the results folder.
    
    Args:
        analysis_folder (str): Path to analysis folder.
        
    Returns:
        str: Path to results folder.
    """
    return os.path.join(analysis_folder, "results")

def read_solution_file(solution_file):
    """
    Read a DICe solution file and return the data as a DataFrame.
    
    Args:
        solution_file (str): Path to solution file.
        
    Returns:
        pandas.DataFrame: DataFrame containing solution data.
    """
    # Read the first few lines to find the header
    with open(solution_file, 'r') as f:
        lines = f.readlines()
    
    # Find the line containing the header (column names)
    header_line = 0
    for i, line in enumerate(lines):
        if 'SUBSET_ID' in line:
            header_line = i
            break
    
    # Read the CSV starting from the header line
    df = pd.read_csv(solution_file, skiprows=header_line)
    
    # Clean up column names by removing any whitespace
    df.columns = df.columns.str.strip()
    
    return df

def parse_camera_log(camera_log_file):
    """
    Parse the Camera_Log.txt file to extract timestamps and temperatures.
    
    Args:
        camera_log_file (str): Path to the camera log file.
        
    Returns:
        tuple: (timestamp_map, temperature_map) - Dictionaries mapping frame indices to timestamps and temperatures.
    """
    timestamp_map = {}
    temperature_map = {}
    
    try:
        if os.path.exists(camera_log_file):
            with open(camera_log_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:  # At least index, timestamp, temperature
                        try:
                            # Format: index, timestamp, temperature, rpm
                            # Handle index with decimal point (e.g., '1.00')
                            index_str = parts[0].strip()
                            if '.' in index_str:
                                # Convert to float first, then to int
                                index = int(float(index_str))
                            else:
                                index = int(index_str)
                            timestamp = float(parts[1].strip())
                            timestamp_map[index] = timestamp
                            
                            # Extract temperature if available
                            if len(parts) >= 3:
                                try:
                                    temperature = float(parts[2].strip())
                                    temperature_map[index] = temperature
                                except (ValueError, IndexError):
                                    # Skip temperature if not valid
                                    pass
                        except (ValueError, IndexError):
                            # Skip invalid lines
                            continue
    except Exception as e:
        print(f"Error parsing camera log file: {str(e)}")
    
    return timestamp_map, temperature_map

def extract_frame_indices_from_xml(notch_folder, cycle, subset, step, vsg):
    """
    Extract the starting frame index and count from the input.xml file.
    
    Args:
        notch_folder (str): Path to notch folder.
        cycle (str): Cycle number.
        subset (str): Subset size.
        step (str): Step size.
        vsg (str): VSG number.
        
    Returns:
        tuple: (start_index, end_index) or (None, None) if not found.
    """
    try:
        # Normalize parameters
        cycle_norm = normalize_parameter(cycle, 5)  # e.g., '00020'
        subset_norm = normalize_parameter(subset, 2) if subset else subset  # e.g., '25'
        step_norm = normalize_parameter(step, 2) if step else step  # e.g., '03'
        vsg_norm = normalize_parameter(vsg, 3) if vsg else vsg  # e.g., '009'

        # Construct the folder name
        folder_name = f"cycle_{cycle_norm}_subset{subset_norm}_step{step_norm}_VSG{vsg_norm}"
        input_xml = os.path.join(notch_folder, "DICe_sequential", folder_name, "input.xml")
        
        if not os.path.exists(input_xml):
            return None, None
        
        # Parse the input.xml file to find the reference image
        start_index = None
        with open(input_xml, 'r') as f:
            content = f.read()
            
            # Look for the reference image parameter
            pattern = r'<Parameter name="reference_image" type="string" value="(.*?)"\s*/>'
            match = re.search(pattern, content)
            
            if match:
                ref_image = match.group(1)
                # Extract the index from the image name
                # Format: Notch_XX_cycle_XXXXX_XXXXX.bmp
                parts = ref_image.split('_')
                if len(parts) >= 5:
                    try:
                        # The last part contains the index and .bmp extension
                        index_part = parts[-1].split('.')[0]
                        start_index = int(index_part)
                    except (ValueError, IndexError):
                        pass
        
        if start_index is not None:
            # Get the results folder
            results_folder = os.path.join(notch_folder, "DICe_sequential", folder_name, "results")
            # Count the number of solution files
            solution_files = get_all_solution_files(results_folder)
            num_frames = len(solution_files)
            
            # Calculate the end index (start_index + num_frames)
            end_index = start_index + num_frames
            
            return start_index, end_index
            
    except Exception as e:
        print(f"Error extracting frame indices from XML: {str(e)}")
    
    return None, None

def get_first_solution_file(results_folder):
    """
    Get the path to the first solution file.
    
    Args:
        results_folder (str): Path to results folder.
        
    Returns:
        str: Path to first solution file.
    """
    solution_files = get_all_solution_files(results_folder)
    if solution_files:
        return solution_files[0]
    else:
        return None

def get_all_solution_files(results_folder):
    """
    Get paths to all solution files.
    
    Args:
        results_folder (str): Path to results folder.
        
    Returns:
        list: List of paths to solution files.
    """
    solution_files = sorted(glob.glob(os.path.join(results_folder, "DICe_solution_*.txt")))
    return solution_files

def extract_frame_number(solution_file):
    """
    Extract the frame number from a solution file name.
    
    Args:
        solution_file (str): Path to solution file.
        
    Returns:
        int: Frame number.
    """
    match = re.search(r'DICe_solution_(\d+)\.txt', os.path.basename(solution_file))
    if match:
        return int(match.group(1))
    return 0

def detect_temp_features(temp_data, time_data, prominence=0.1, width=5):
    """
    Detect peaks and valleys in temperature data.
    
    Args:
        temp_data (list): List of temperature values.
        time_data (list): List of corresponding time values.
        prominence (float): Minimum prominence of peaks/valleys to detect.
        width (int): Minimum width of peaks/valleys to detect.
        
    Returns:
        tuple: (peaks, valleys) where each is a tuple of (times, temps) for detected features.
    """
    # Check if scipy is installed
    try:
        import numpy as np
        from scipy.signal import find_peaks
    except ImportError:
        # If scipy is not installed, return empty results
        print("SciPy module is not installed. Temperature feature detection will be disabled.")
        return ([], []), ([], [])
    
    # Convert data to numpy arrays if they aren't already
    temp_array = np.array(temp_data)
    time_array = np.array(time_data)
    
    # Find peaks (maximums)
    if len(temp_array) > width*2:
        peaks, _ = find_peaks(temp_array, prominence=prominence, width=width)
        peak_times = time_array[peaks] if peaks.size > 0 else []
        peak_temps = temp_array[peaks] if peaks.size > 0 else []
    else:
        peak_times = []
        peak_temps = []
    
    # Find valleys (minimums) by inverting the data
    if len(temp_array) > width*2:
        valleys, _ = find_peaks(-temp_array, prominence=prominence, width=width)
        valley_times = time_array[valleys] if valleys.size > 0 else []
        valley_temps = temp_array[valleys] if valleys.size > 0 else []
    else:
        valley_times = []
        valley_temps = []
    
    return (peak_times, peak_temps), (valley_times, valley_temps)


def calculate_principal_strains(strain_xx, strain_yy, shear_xy):
    """
    Calculate principal strains from strain components.
    
    Args:
        strain_xx (float): Strain in X direction.
        strain_yy (float): Strain in Y direction.
        shear_xy (float): Shear strain.
        
    Returns:
        tuple: (max_strain, min_strain)
    """
    avg = (strain_xx + strain_yy) / 2
    diff = (strain_xx - strain_yy) / 2
    radius = np.sqrt(diff * diff + (shear_xy / 2) * (shear_xy / 2))
    
    max_strain = avg + radius  # Maximum principal strain (e1)
    min_strain = avg - radius  # Minimum principal strain (e2)
    
    return max_strain, min_strain

def generate_point_map(notch_folder, analysis_folder, logger, existing_points=None):
    """
    Generate a point map for a notch folder that displays subset IDs for selection.
    Allows selecting multiple points and deselecting points.
    
    Args:
        notch_folder (str): Path to notch folder.
        analysis_folder (str): Path to analysis folder.
        logger (logging.Logger): Logger object.
        existing_points (list): List of already selected points (optional).
        
    Returns:
        list: List of selected subset IDs or empty list if canceled.
    """
    notch_name = os.path.basename(notch_folder)
    results_folder = get_results_folder(analysis_folder)
    first_solution_file = get_first_solution_file(results_folder)
    
    if first_solution_file is None:
        logger.error(f"No solution file found for {notch_name}")
        return []
    
    try:
        # Read solution data
        df = read_solution_file(first_solution_file)
        
        # Extract subset coordinates and IDs
        subset_ids = df['SUBSET_ID'].astype(int).values
        x_coords = df['COORDINATE_X'].values
        y_coords = df['COORDINATE_Y'].values
        
        # Initialize existing points if not provided
        if existing_points is None:
            existing_points = []
        
        # Create a scatter plot of the subset points
        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(x_coords, y_coords, c='blue', s=10)
        
        # Add labels to points with their IDs
        label_step = max(1, len(subset_ids) // 100)  # Show only a subset of labels to avoid overcrowding
        for i in range(0, len(subset_ids), label_step):
            ax.annotate(f"{subset_ids[i]}", (x_coords[i], y_coords[i]), fontsize=8)
        
        ax.set_title(f"Point Map for {notch_name}\nClick to select/deselect points - Multiple points allowed")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        
        # Create a class to handle picking points
        class PointPicker:
            def __init__(self):
                self.selected_ids = existing_points.copy()  # Start with existing points
                self.highlighted_points = []  # Track highlighted points for cleanup
                self.point_labels = []  # Track point labels for cleanup
                self.cid = fig.canvas.mpl_connect('button_press_event', self.on_click)
                
                # Highlight existing points
                self.highlight_existing_points()
            
            def highlight_existing_points(self):
                # Highlight any existing points
                for point_id in self.selected_ids:
                    # Find the index of this point_id
                    try:
                        idx = np.where(subset_ids == point_id)[0][0]
                        self.highlight_point(idx)
                    except (IndexError, TypeError):
                        # Point not found, skip it
                        pass
            
            def highlight_point(self, idx):
                # Add a red circle around the point
                highlight = ax.plot(x_coords[idx], y_coords[idx], 'ro', markersize=15, fillstyle='none')[0]
                self.highlighted_points.append(highlight)
                
                # Add a label with the ID
                label = ax.annotate(f"ID: {subset_ids[idx]}", (x_coords[idx], y_coords[idx]),
                                    fontsize=12, color='red', fontweight='bold',
                                    xytext=(10, 10), textcoords='offset points',
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red"))
                self.point_labels.append(label)
                plt.draw()
            
            def remove_highlight(self, idx):
                # Find and remove the highlight and label for this point
                point_id = subset_ids[idx]
                # We need to rebuild the highlights and labels
                for h in self.highlighted_points:
                    h.remove()
                for l in self.point_labels:
                    l.remove()
                
                self.highlighted_points = []
                self.point_labels = []
                
                # Rehighlight all selected points except the one we're removing
                for existing_id in self.selected_ids:
                    try:
                        existing_idx = np.where(subset_ids == existing_id)[0][0]
                        self.highlight_point(existing_idx)
                    except (IndexError, TypeError):
                        pass
                
                plt.draw()
            
            def on_click(self, event):
                if event.inaxes != ax:
                    return
                
                # Find closest point to click
                dist = np.sqrt((x_coords - event.xdata)**2 + (y_coords - event.ydata)**2)
                idx = np.argmin(dist)
                point_id = subset_ids[idx]
                
                # Toggle selection status
                if point_id in self.selected_ids:
                    # Deselect the point
                    self.selected_ids.remove(point_id)
                    # Need to remove highlight
                    self.remove_highlight(idx)
                    status_text.set_text(f"Deselected point ID: {point_id}")
                else:
                    # Select the point
                    self.selected_ids.append(point_id)
                    # Highlight the selected point
                    self.highlight_point(idx)
                    status_text.set_text(f"Selected point ID: {point_id}")
                
                count_text.set_text(f"Selected {len(self.selected_ids)} points: {self.selected_ids}")
                plt.draw()
        
        # Add a status bar to show selection state
        status_text = ax.text(0.05, 0.02, "Click on points to select/deselect", transform=ax.transAxes,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add a counter for selected points
        count_text = ax.text(0.05, 0.06, f"Selected {len(existing_points)} points: {existing_points}", 
                            transform=ax.transAxes,
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # Create button to save selection
        save_ax = plt.axes([0.7, 0.01, 0.1, 0.04])
        save_button = Button(save_ax, 'Save')
        
        # Create button to cancel
        cancel_ax = plt.axes([0.81, 0.01, 0.1, 0.04])
        cancel_button = Button(cancel_ax, 'Cancel')
        
        # Create button to clear all selections
        clear_ax = plt.axes([0.59, 0.01, 0.1, 0.04])
        clear_button = Button(clear_ax, 'Clear All')
        
        picker = PointPicker()
        
        # Define button click callbacks
        def save_callback(event):
            plt.close()
        save_button.on_clicked(save_callback)
        
        def cancel_callback(event):
            picker.selected_ids = existing_points.copy()  # Restore original selection
            plt.close()
        cancel_button.on_clicked(cancel_callback)
        
        def clear_callback(event):
            picker.selected_ids = []
            for h in picker.highlighted_points:
                h.remove()
            for l in picker.point_labels:
                l.remove()
            picker.highlighted_points = []
            picker.point_labels = []
            status_text.set_text("Cleared all selections")
            count_text.set_text("Selected 0 points: []")
            plt.draw()
        clear_button.on_clicked(clear_callback)
        
        # Show the plot
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()
        
        return picker.selected_ids
    
    except Exception as e:
        logger.error(f"Error generating point map for {notch_name}: {str(e)}")
        return []

def validate_point_data(df, subset_id, gamma_threshold, sigma_threshold, logger):
    """
    Validate point data against thresholds.
    
    Args:
        df (pandas.DataFrame): DataFrame containing solution data.
        subset_id (int): Subset ID to validate.
        gamma_threshold (float): Threshold for gamma validation.
        sigma_threshold (float): Threshold for sigma validation.
        logger (logging.Logger): Logger object.
        
    Returns:
        bool: True if valid, False otherwise.
    """
    row = df[df['SUBSET_ID'] == subset_id]
    
    if len(row) == 0:
        logger.warning(f"Subset ID {subset_id} not found in solution file")
        return False
    
    row = row.iloc[0]
    
    # Check status flag
    status_flag = int(row['STATUS_FLAG'])
    if status_flag != STATUS_FLAG_OK:
        logger.warning(f"Status flag for subset ID {subset_id} is {status_flag}, expected {STATUS_FLAG_OK}")
        return False
    
    # Check gamma value
    gamma = float(row['GAMMA'])
    if gamma > gamma_threshold:
        logger.warning(f"Gamma value for subset ID {subset_id} is {gamma}, exceeds threshold {gamma_threshold}")
        return False
    
    # Check sigma value
    sigma = float(row['SIGMA'])
    if sigma > sigma_threshold:
        logger.warning(f"Sigma value for subset ID {subset_id} is {sigma}, exceeds threshold {sigma_threshold}")
        return False
    
    return True

def process_strain_data(notch_folder, analysis_folder, subset_id, gamma_threshold, sigma_threshold, logger, timestamp_map=None, temperature_map=None, frame_index_range=None):
    """
    Process strain data for a single notch folder.
    
    Args:
        notch_folder (str): Path to notch folder.
        analysis_folder (str): Path to analysis folder.
        subset_id (int): Subset ID to analyze.
        gamma_threshold (float): Threshold for gamma validation.
        sigma_threshold (float): Threshold for sigma validation.
        logger (logging.Logger): Logger object.
        timestamp_map (dict, optional): Dictionary mapping frame indices to timestamps.
        frame_index_range (tuple, optional): Tuple of (start_index, end_index) for the frames in the cycle.
        
    Returns:
        pandas.DataFrame or None: DataFrame containing strain data or None if error.
    """
    notch_name = os.path.basename(notch_folder)
    results_folder = get_results_folder(analysis_folder)
    solution_files = get_all_solution_files(results_folder)
    
    if not solution_files:
        logger.error(f"No solution files found for {notch_name}")
        return None
    
    try:
        # Initialize lists to store data
        frames = []
        times = []
        strain_xx_values = []
        strain_yy_values = []
        strain_xy_values = []
        max_strain_values = []
        min_strain_values = []
        gamma_values = []
        sigma_values = []
        status_values = []
        valid_flags = []
        
        # Collect raw timestamps and temperatures first
        raw_timestamps = []
        raw_temperatures = []
        frame_data = []
        
        # Process each solution file
        for solution_file in solution_files:
            frame_num = extract_frame_number(solution_file)
            df = read_solution_file(solution_file)
            
            # Find the row with our subset ID
            row = df[df['SUBSET_ID'] == subset_id]
            if len(row) == 0:
                continue
            
            row = row.iloc[0]
            
            # Extract strain components
            strain_xx = row['VSG_STRAIN_XX']
            strain_yy = row['VSG_STRAIN_YY']
            strain_xy = row['VSG_STRAIN_XY']
            gamma = row['GAMMA']
            sigma = row['SIGMA']
            status = row['STATUS_FLAG']
            
            # Calculate principal strains
            max_strain, min_strain = calculate_principal_strains(strain_xx, strain_yy, strain_xy)
            
            # Validate point data
            valid = (status == STATUS_FLAG_OK and 
                     gamma <= gamma_threshold and 
                     sigma <= sigma_threshold)
            
            # Determine raw timestamp and temperature
            raw_time = frame_num  # Default to frame number
            raw_temp = None        # Default to no temperature data
            
            if timestamp_map and frame_index_range:
                start_index, end_index = frame_index_range
                # Calculate the actual index in the Camera_Log.txt file
                actual_index = start_index + frame_num - 1  # Adjust for 0-based frame_num
                raw_time = timestamp_map.get(actual_index, frame_num)
                
                # Get temperature data if available
                if temperature_map:
                    raw_temp = temperature_map.get(actual_index, None)
            
            # Store all the data in temporary collections
            frame_data.append({
                'frame_num': frame_num,
                'raw_time': raw_time,
                'temperature': raw_temp,
                'strain_xx': strain_xx,
                'strain_yy': strain_yy,
                'strain_xy': strain_xy,
                'max_strain': max_strain,
                'min_strain': min_strain,
                'gamma': gamma,
                'sigma': sigma,
                'status': status,
                'valid': valid
            })
            
            raw_timestamps.append(raw_time)
            if raw_temp is not None:
                raw_temperatures.append(raw_temp)
        
        # Sort by frame number
        frame_data.sort(key=lambda x: x['frame_num'])
        
        # Get the first timestamp to normalize all others
        if raw_timestamps and timestamp_map and frame_index_range:
            first_timestamp = min(raw_timestamps)
        else:
            first_timestamp = 0
        
        # Now populate the arrays with normalized timestamps
        temperatures = []
        for data in frame_data:
            frames.append(data['frame_num'])
            # Normalize the timestamp by subtracting the first timestamp
            times.append(data['raw_time'] - first_timestamp)
            temperatures.append(data['temperature'])  # This might be None for some frames
            strain_xx_values.append(data['strain_xx'])
            strain_yy_values.append(data['strain_yy'])
            strain_xy_values.append(data['strain_xy'])
            max_strain_values.append(data['max_strain'])
            min_strain_values.append(data['min_strain'])
            gamma_values.append(data['gamma'])
            sigma_values.append(data['sigma'])
            status_values.append(data['status'])
            valid_flags.append(data['valid'])
        
        # Create DataFrame from collected data
        data = {
            'Frame': frames,
            'Time': times,
            'Temperature': temperatures,
            'Exx': strain_xx_values,
            'Eyy': strain_yy_values,
            'Exy': strain_xy_values,
            'MaxStrain': max_strain_values,
            'MinStrain': min_strain_values,
            'Gamma': gamma_values,
            'Sigma': sigma_values,
            'Status': status_values,
            'Valid': valid_flags
        }
        
        # Return as DataFrame
        return pd.DataFrame(data)
    
    except Exception as e:
        logger.error(f"Error processing strain data for {notch_name}: {str(e)}")
        return None

def create_strain_plots(strain_data_dict, output_dir, settings, logger):
    """
    Create plots comparing strain data across all notches, with multiple points per notch.
    
    Args:
        strain_data_dict (dict): Dictionary with structure {notch_name: {point_id: DataFrame}}.
        output_dir (str): Directory to save plots.
        settings (dict): Settings dictionary.
        logger (logging.Logger): Logger object.
        
    Returns:
        tuple: (max_strain_plots, min_strain_plots) dictionaries with notch-specific plots
    """
    # Check if we're using real timestamps instead of frame numbers
    use_timestamps = settings.get('use_timestamps', False) and settings.get('camera_log_file', '')
    # Create output directory if it doesn't exist
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get analysis parameters to include in filenames
    cycle = settings['cycle']
    subset = settings['subset']
    step = settings['step']
    vsg = settings['vsg']
    
    # Use normalized values for consistent naming
    cycle_norm = normalize_parameter(cycle, 5)
    subset_norm = normalize_parameter(subset, 2)
    step_norm = normalize_parameter(step, 2)
    vsg_norm = normalize_parameter(vsg, 3)
    param_str = f"c{cycle_norm}_s{subset_norm}_st{step_norm}_vsg{vsg_norm}"
    
    # Color palette for notches and points
    notch_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    # Lists to track plot paths
    max_strain_plots = {}
    min_strain_plots = {}
    
    try:
        # Create combined plots for all notches (across different points)
        fig_max_all = go.Figure()
        fig_min_all = go.Figure()
        
        # Process each notch with potentially multiple points
        for notch_idx, (notch_name, points_data) in enumerate(strain_data_dict.items()):
            notch_color = notch_colors[notch_idx % len(notch_colors)]
            
            # Also create a combined plot for all points in this notch
            fig_max_notch_combined = go.Figure()
            fig_min_notch_combined = go.Figure()
            
            # Process each point for this notch
            for point_id, df in points_data.items():
                # Create point-specific plots
                fig_max_point = go.Figure()
                fig_min_point = go.Figure()
                
                # Create a display name that includes both notch and point ID
                display_name = f"{notch_name} (Point {point_id})"
                
                # Add traces to the point-specific plots
                fig_max_point.add_trace(go.Scatter(
                    x=df['Time'],
                    y=df['MaxStrain'],
                    name=display_name,
                    line=dict(color=notch_color),
                    mode='lines',
                    visible=True
                ))
                
                # Add temperature overlay if enabled and temperature data exists
                overlay_temperature = settings.get('overlay_temperature', False)
                has_temperature_data = 'Temperature' in df.columns and not df['Temperature'].isnull().all()
                
                if overlay_temperature and has_temperature_data:
                    # Filter out None values from temperature data
                    valid_temp_mask = df['Temperature'].notna()
                    valid_times = df.loc[valid_temp_mask, 'Time'].tolist()
                    valid_temps = df.loc[valid_temp_mask, 'Temperature'].tolist()
                    
                    # Add secondary y-axis for temperature
                    fig_max_point.add_trace(go.Scatter(
                        x=valid_times,
                        y=valid_temps,
                        name=f"Temperature",
                        line=dict(color='red', dash='dash'),
                        mode='lines',
                        yaxis="y2",  # Use secondary y-axis
                    ))
                    
                    # Detect temperature peaks and valleys if feature detection is enabled
                    detect_features = settings.get('detect_temp_features', True)
                    if detect_features and len(valid_temps) > 10:  # Only detect features if we have enough data points
                        try:
                            # Check if scipy is available before attempting feature detection
                            try:
                                import scipy
                                has_scipy = True
                            except ImportError:
                                has_scipy = False
                                logger.info("SciPy package not installed. Temperature feature detection disabled.")
                            
                            if has_scipy:
                                # Calculate prominence as a percentage of the temperature range
                                temp_range = max(valid_temps) - min(valid_temps)
                                prominence = max(0.05 * temp_range, 0.1)  # At least 5% of the range, minimum 0.1
                                
                                # Detect peaks and valleys
                                (peak_times, peak_temps), (valley_times, valley_temps) = detect_temp_features(
                                    valid_temps, valid_times, prominence=prominence, width=3
                                )
                                
                                # Add markers for peaks
                                if len(peak_times) > 0:
                                    fig_max_point.add_trace(go.Scatter(
                                        x=peak_times,
                                        y=peak_temps,
                                        name="Temp Peaks",
                                        mode='markers',
                                        marker=dict(color='red', size=10, symbol='triangle-up'),
                                        yaxis="y2"
                                    ))
                                
                                # Add markers for valleys
                                if len(valley_times) > 0:
                                    fig_max_point.add_trace(go.Scatter(
                                        x=valley_times,
                                        y=valley_temps,
                                        name="Temp Valleys",
                                        mode='markers',
                                        marker=dict(color='blue', size=10, symbol='triangle-down'),
                                        yaxis="y2"
                                    ))
                        except Exception as e:
                            # If feature detection fails, just log the error and continue without markers
                            logger.warning(f"Failed to detect temperature features: {str(e)}")
                            pass
                
                # Update layout based on whether using timestamps or frame numbers
                x_axis_title = "Time" if use_timestamps else "Frame Number"
                
                # Configure layout with secondary y-axis for temperature if needed
                layout_config = {
                    "title": f"Maximum Principal Strain - {notch_name}, Point {point_id}",
                    "xaxis_title": x_axis_title,
                    "yaxis_title": "Maximum Principal Strain",
                    "hovermode": "closest"
                }
                
                # Add secondary y-axis configuration if temperature overlay is enabled
                if overlay_temperature and has_temperature_data:
                    layout_config["yaxis2"] = {
                        "title": "Temperature (°F)",
                        "overlaying": "y",
                        "side": "right",
                        "showgrid": False
                    }
                
                fig_max_point.update_layout(**layout_config)
                
                fig_min_point.add_trace(go.Scatter(
                    x=df['Time'],
                    y=df['MinStrain'],
                    name=display_name,
                    line=dict(color=notch_color),
                    mode='lines',
                    visible=True
                ))
                
                # Add temperature overlay to minimum strain plots if enabled
                if overlay_temperature and has_temperature_data:
                    # Filter out None values from temperature data
                    valid_temp_mask = df['Temperature'].notna()
                    valid_times = df.loc[valid_temp_mask, 'Time'].tolist()
                    valid_temps = df.loc[valid_temp_mask, 'Temperature'].tolist()
                    
                    # Add secondary y-axis for temperature
                    fig_min_point.add_trace(go.Scatter(
                        x=valid_times,
                        y=valid_temps,
                        name=f"Temperature",
                        line=dict(color='red', dash='dash'),
                        mode='lines',
                        yaxis="y2",  # Use secondary y-axis
                    ))
                    
                    # Detect temperature peaks and valleys if feature detection is enabled
                    detect_features = settings.get('detect_temp_features', True)
                    if detect_features and len(valid_temps) > 10:  # Only detect features if we have enough data points
                        try:
                            # Check if scipy is available before attempting feature detection
                            try:
                                import scipy
                                has_scipy = True
                            except ImportError:
                                has_scipy = False
                                logger.info("SciPy package not installed. Temperature feature detection disabled.")
                            
                            if has_scipy:
                                # Calculate prominence as a percentage of the temperature range
                                temp_range = max(valid_temps) - min(valid_temps)
                                prominence = max(0.05 * temp_range, 0.1)  # At least 5% of the range, minimum 0.1
                                
                                # Detect peaks and valleys
                                (peak_times, peak_temps), (valley_times, valley_temps) = detect_temp_features(
                                    valid_temps, valid_times, prominence=prominence, width=3
                                )
                                
                                # Add markers for peaks
                                if len(peak_times) > 0:
                                    fig_min_point.add_trace(go.Scatter(
                                        x=peak_times,
                                        y=peak_temps,
                                        name="Temp Peaks",
                                        mode='markers',
                                        marker=dict(color='red', size=10, symbol='triangle-up'),
                                        yaxis="y2"
                                    ))
                                
                                # Add markers for valleys
                                if len(valley_times) > 0:
                                    fig_min_point.add_trace(go.Scatter(
                                        x=valley_times,
                                        y=valley_temps,
                                        name="Temp Valleys",
                                        mode='markers',
                                        marker=dict(color='blue', size=10, symbol='triangle-down'),
                                        yaxis="y2"
                                    ))
                        except Exception as e:
                            # If feature detection fails, just log the error and continue without markers
                            logger.warning(f"Failed to detect temperature features: {str(e)}")
                            pass
                
                # Update layout for minimum strain plot
                layout_config = {
                    "title": f"Minimum Principal Strain - {notch_name}, Point {point_id}",
                    "xaxis_title": x_axis_title,
                    "yaxis_title": "Minimum Principal Strain",
                    "hovermode": "closest"
                }
                
                # Add secondary y-axis configuration if temperature overlay is enabled
                if overlay_temperature and has_temperature_data:
                    layout_config["yaxis2"] = {
                        "title": "Temperature (°F)",
                        "overlaying": "y",
                        "side": "right",
                        "showgrid": False
                    }
                
                fig_min_point.update_layout(**layout_config)
                
                # Add traces to the notch-combined plots
                fig_max_notch_combined.add_trace(go.Scatter(
                    x=df['Time'],
                    y=df['MaxStrain'],
                    name=f"Point {point_id}",
                    line=dict(color=notch_color, dash='solid' if point_id % 2 == 0 else 'dash'),
                    mode='lines',
                    visible=True
                ))
                
                fig_min_notch_combined.add_trace(go.Scatter(
                    x=df['Time'],
                    y=df['MinStrain'],
                    name=f"Point {point_id}",
                    line=dict(color=notch_color, dash='solid' if point_id % 2 == 0 else 'dash'),
                    mode='lines',
                    visible=True
                ))
                
                # Add traces to the all-notches plots
                fig_max_all.add_trace(go.Scatter(
                    x=df['Time'],
                    y=df['MaxStrain'],
                    name=display_name,
                    line=dict(color=notch_color, dash='solid' if point_id % 2 == 0 else 'dash'),
                    mode='lines',
                    visible=True
                ))
                
                fig_min_all.add_trace(go.Scatter(
                    x=df['Time'],
                    y=df['MinStrain'],
                    name=display_name,
                    line=dict(color=notch_color, dash='solid' if point_id % 2 == 0 else 'dash'),
                    mode='lines',
                    visible=True
                ))
                
                # Update point-specific plot layouts
                # Configure the layout based on whether temperature is overlaid
                if overlay_temperature and has_temperature_data:
                    fig_max_point.update_layout(
                        title=f"{notch_name} - Point {point_id} - Maximum Principal Strain vs. Time",
                        xaxis_title=x_axis_title,
                        yaxis_title="Maximum Principal Strain (e1)",
                        yaxis2=dict(
                            title=dict(text="Temperature (°F)", font=dict(color="red")),
                            tickfont=dict(color="red"),
                            anchor="x",
                            overlaying="y",
                            side="right"
                        ),
                        hovermode='x unified',
                        showlegend=True,
                        width=1000,
                        height=600
                    )
                else:
                    fig_max_point.update_layout(
                        title=f"{notch_name} - Point {point_id} - Maximum Principal Strain vs. Time",
                        xaxis_title=x_axis_title,
                        yaxis_title="Maximum Principal Strain (e1)",
                        hovermode='x unified',
                        showlegend=True,
                        width=1000,
                        height=600
                    )
                
                # Configure the layout based on whether temperature is overlaid
                if overlay_temperature and has_temperature_data:
                    fig_min_point.update_layout(
                        title=f"{notch_name} - Point {point_id} - Minimum Principal Strain vs. Time",
                        xaxis_title=x_axis_title,
                        yaxis_title="Minimum Principal Strain (e2)",
                        yaxis2=dict(
                            title=dict(text="Temperature (°F)", font=dict(color="red")),
                            tickfont=dict(color="red"),
                            anchor="x",
                            overlaying="y",
                            side="right"
                        ),
                        hovermode='x unified',
                        showlegend=True,
                        width=1000,
                        height=600
                    )
                else:
                    fig_min_point.update_layout(
                        title=f"{notch_name} - Point {point_id} - Minimum Principal Strain vs. Time",
                        xaxis_title=x_axis_title,
                        yaxis_title="Minimum Principal Strain (e2)",
                        hovermode='x unified',
                        showlegend=True,
                        width=1000,
                        height=600
                    )
                
                # Add grid to point-specific plots
                for fig in [fig_max_point, fig_min_point]:
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                
                # Save point-specific plots
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Format: max_strain_Notch_01_point123_c00020_s25_st03_vsg009_20250519_123456.html
                point_key = f"{notch_name}_point{point_id}"
                
                if settings.get('interactive_plots', True):
                    max_strain_html = os.path.join(plots_dir, f"max_strain_{notch_name}_point{point_id}_{param_str}_{timestamp}.html")
                    min_strain_html = os.path.join(plots_dir, f"min_strain_{notch_name}_point{point_id}_{param_str}_{timestamp}.html")
                    
                    fig_max_point.write_html(max_strain_html)
                    fig_min_point.write_html(min_strain_html)
                    
                    max_strain_plots[point_key] = {'html': max_strain_html}
                    min_strain_plots[point_key] = {'html': min_strain_html}
                    
                    logger.info(f"Interactive strain plots saved for {notch_name}, point {point_id}")
                
                if settings.get('static_plots', True):
                    max_strain_png = os.path.join(plots_dir, f"max_strain_{notch_name}_point{point_id}_{param_str}_{timestamp}.png")
                    min_strain_png = os.path.join(plots_dir, f"min_strain_{notch_name}_point{point_id}_{param_str}_{timestamp}.png")
                    
                    fig_max_point.write_image(max_strain_png)
                    fig_min_point.write_image(min_strain_png)
                    
                    if point_key in max_strain_plots:
                        max_strain_plots[point_key]['png'] = max_strain_png
                    else:
                        max_strain_plots[point_key] = {'png': max_strain_png}
                        
                    if point_key in min_strain_plots:
                        min_strain_plots[point_key]['png'] = min_strain_png
                    else:
                        min_strain_plots[point_key] = {'png': min_strain_png}
                    
                    logger.info(f"Static strain plots saved for {notch_name}, point {point_id}")
            
            # Only create combined notch plots if there are multiple points for this notch
            if len(points_data) > 1:
                # Update notch-combined plot layouts
                # Update title based on whether using timestamps or frame numbers
                title_suffix = "vs. Time" if use_timestamps else "vs. Frame"
                
                # Configure max strain plot for notch
                max_notch_layout = {
                    "title": f"{notch_name} - All Points - Maximum Principal Strain {title_suffix}",
                    "xaxis_title": x_axis_title,
                    "yaxis_title": "Maximum Principal Strain (e1)",
                    "hovermode": 'x unified',
                    "showlegend": True,
                    "width": 1000,
                    "height": 600
                }
                
                # Configure min strain plot for notch
                min_notch_layout = {
                    "title": f"{notch_name} - All Points - Minimum Principal Strain {title_suffix}",
                    "xaxis_title": x_axis_title,
                    "yaxis_title": "Minimum Principal Strain (e2)",
                    "hovermode": 'x unified',
                    "showlegend": True,
                    "width": 1000,
                    "height": 600
                }
                
                # Add secondary y-axis configuration if temperature overlay is enabled
                if overlay_temperature and has_temperature_data:
                    temp_axis_config = {
                        "title": dict(text="Temperature (°F)", font=dict(color="red")),
                        "tickfont": dict(color="red"),
                        "anchor": "x",
                        "overlaying": "y",
                        "side": "right"
                    }
                    max_notch_layout["yaxis2"] = temp_axis_config
                    min_notch_layout["yaxis2"] = temp_axis_config
                
                fig_max_notch_combined.update_layout(**max_notch_layout)
                fig_min_notch_combined.update_layout(**min_notch_layout)
                
                # Add grid to notch-combined plots
                for fig in [fig_max_notch_combined, fig_min_notch_combined]:
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                
                # Save notch-combined plots
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if settings.get('interactive_plots', True):
                    max_notch_html = os.path.join(plots_dir, f"max_strain_{notch_name}_all_points_{param_str}_{timestamp}.html")
                    min_notch_html = os.path.join(plots_dir, f"min_strain_{notch_name}_all_points_{param_str}_{timestamp}.html")
                    
                    fig_max_notch_combined.write_html(max_notch_html)
                    fig_min_notch_combined.write_html(min_notch_html)
                    
                    max_strain_plots[f"{notch_name}_all_points"] = {'html': max_notch_html}
                    min_strain_plots[f"{notch_name}_all_points"] = {'html': min_notch_html}
                    
                    logger.info(f"Combined interactive strain plots saved for all points in {notch_name}")
                
                if settings.get('static_plots', True):
                    max_notch_png = os.path.join(plots_dir, f"max_strain_{notch_name}_all_points_{param_str}_{timestamp}.png")
                    min_notch_png = os.path.join(plots_dir, f"min_strain_{notch_name}_all_points_{param_str}_{timestamp}.png")
                    
                    fig_max_notch_combined.write_image(max_notch_png)
                    fig_min_notch_combined.write_image(min_notch_png)
                    
                    key = f"{notch_name}_all_points"
                    if key in max_strain_plots:
                        max_strain_plots[key]['png'] = max_notch_png
                    else:
                        max_strain_plots[key] = {'png': max_notch_png}
                        
                    if key in min_strain_plots:
                        min_strain_plots[key]['png'] = min_notch_png
                    else:
                        min_strain_plots[key] = {'png': min_notch_png}
                    
                    logger.info(f"Combined static strain plots saved for all points in {notch_name}")
        
        # Only create all-notches plots if we have more than one notch
        if len(strain_data_dict) > 1:
            # Update all-notches plot layouts
            # Update title based on whether using timestamps or frame numbers
            title_suffix = "vs. Time" if use_timestamps else "vs. Frame"
            
            # Configure max strain plot for all notches
            max_all_layout = {
                "title": f"Maximum Principal Strain {title_suffix} (All Notches)",
                "xaxis_title": x_axis_title,
                "yaxis_title": "Maximum Principal Strain (e1)",
                "hovermode": 'x unified',
                "showlegend": True,
                "width": 1200,
                "height": 800
            }
            
            # Configure min strain plot for all notches
            min_all_layout = {
                "title": f"Minimum Principal Strain {title_suffix} (All Notches)",
                "xaxis_title": x_axis_title,
                "yaxis_title": "Minimum Principal Strain (e2)",
                "hovermode": 'x unified',
                "showlegend": True,
                "width": 1200,
                "height": 800
            }
            
            # Add secondary y-axis configuration if temperature overlay is enabled and any notch has temperature data
            if overlay_temperature and any(has_temperature_data for notch_name, points_data in strain_data_dict.items() 
                                            for point_id, df in points_data.items() if 'Temperature' in df.columns):
                temp_axis_config = {
                    "title": dict(text="Temperature (°F)", font=dict(color="red")),
                    "tickfont": dict(color="red"),
                    "anchor": "x",
                    "overlaying": "y",
                    "side": "right"
                }
                max_all_layout["yaxis2"] = temp_axis_config
                min_all_layout["yaxis2"] = temp_axis_config
            
            fig_max_all.update_layout(**max_all_layout)
            fig_min_all.update_layout(**min_all_layout)
            
            # Add grid to all-notches plots
            for fig in [fig_max_all, fig_min_all]:
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            
            # Save all-notches plots
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if settings.get('interactive_plots', True):
                max_all_html = os.path.join(plots_dir, f"max_strain_all_notches_{param_str}_{timestamp}.html")
                min_all_html = os.path.join(plots_dir, f"min_strain_all_notches_{param_str}_{timestamp}.html")
                
                fig_max_all.write_html(max_all_html)
                fig_min_all.write_html(min_all_html)
                
                max_strain_plots['all_notches'] = {'html': max_all_html}
                min_strain_plots['all_notches'] = {'html': min_all_html}
                
                logger.info(f"Combined interactive strain plots saved for all notches")
            
            if settings.get('static_plots', True):
                max_all_png = os.path.join(plots_dir, f"max_strain_all_notches_{param_str}_{timestamp}.png")
                min_all_png = os.path.join(plots_dir, f"min_strain_all_notches_{param_str}_{timestamp}.png")
                
                fig_max_all.write_image(max_all_png)
                fig_min_all.write_image(min_all_png)
                
                if 'all_notches' in max_strain_plots:
                    max_strain_plots['all_notches']['png'] = max_all_png
                else:
                    max_strain_plots['all_notches'] = {'png': max_all_png}
                    
                if 'all_notches' in min_strain_plots:
                    min_strain_plots['all_notches']['png'] = min_all_png
                else:
                    min_strain_plots['all_notches'] = {'png': min_all_png}
                
                logger.info(f"Combined static strain plots saved for all notches")
        
        logger.info(f"All strain plots saved to {plots_dir}")
        return max_strain_plots, min_strain_plots
    
    except Exception as e:
        logger.error(f"Error creating strain plots: {str(e)}")
        return {}, {}

def save_strain_data_to_csv(strain_data_dict, output_dir, settings, logger):
    """
    Save strain data to CSV files.
    
    Args:
        strain_data_dict (dict): Dictionary mapping notch names to DataFrames.
        output_dir (str): Directory to save CSV files.
        settings (dict): Settings dictionary.
        logger (logging.Logger): Logger object.
        
    Returns:
        str: Path to data directory.
    """
    try:
        # Create data directory if it doesn't exist
        data_dir = os.path.join(output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Get analysis parameters to include in filenames
        cycle = settings['cycle']
        subset = settings['subset']
        step = settings['step']
        vsg = settings['vsg']
        
        # Use normalized values for consistent naming
        cycle_norm = normalize_parameter(cycle, 5)
        subset_norm = normalize_parameter(subset, 2)
        step_norm = normalize_parameter(step, 2)
        vsg_norm = normalize_parameter(vsg, 3)
        param_str = f"c{cycle_norm}_s{subset_norm}_st{step_norm}_vsg{vsg_norm}"
        
        # Save each notch's data to a separate CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a combined data frame for all notches and points
        combined_data = pd.DataFrame()
        summary_data = []
        
        # Iterate through the nested dictionary: {notch_name: {point_id: DataFrame}}
        for notch_name, point_dict in strain_data_dict.items():
            for point_id, df in point_dict.items():
                # Save individual point data to CSV
                csv_file = os.path.join(data_dir, f"{notch_name}_point{point_id}_strain_data_{param_str}_{timestamp}.csv")
                df.to_csv(csv_file, index=False)
                logger.info(f"Saved strain data for {notch_name}, point {point_id} to {csv_file}")
                
                # Process for combined data
                notch_df = df.copy()
                notch_df['Notch'] = notch_name
                notch_df['PointID'] = point_id
                columns_to_keep = ['Notch', 'PointID', 'Frame', 'Time', 'MaxStrain', 'MinStrain', 'Valid']
                available_cols = [col for col in columns_to_keep if col in notch_df.columns]
                notch_df = notch_df[available_cols]
                combined_data = pd.concat([combined_data, notch_df])
                
                # Add summary statistics
                if 'MaxStrain' in df.columns and 'MinStrain' in df.columns:
                    valid_data = df[df['Valid'] == True] if 'Valid' in df.columns else df
                    if not valid_data.empty:
                        max_strain = valid_data['MaxStrain'].max()
                        min_strain = valid_data['MinStrain'].min()
                        summary_data.append({
                            'Notch': notch_name,
                            'PointID': point_id,
                            'MaxStrainPeak': max_strain,
                            'MinStrainPeak': min_strain
                        })
            
            # Also save a combined CSV for all points in this notch
            notch_combined = pd.DataFrame()
            for point_id, df in point_dict.items():
                point_df = df.copy()
                point_df['PointID'] = point_id
                notch_combined = pd.concat([notch_combined, point_df])
            
            if not notch_combined.empty:
                notch_csv = os.path.join(data_dir, f"{notch_name}_all_points_strain_data_{param_str}_{timestamp}.csv")
                notch_combined.to_csv(notch_csv, index=False)
                logger.info(f"Saved combined strain data for all points in {notch_name} to {notch_csv}")
        
        combined_csv = os.path.join(data_dir, f"combined_strain_data_{param_str}_{timestamp}.csv")
        combined_data.to_csv(combined_csv, index=False)
        logger.info(f"Saved combined strain data to {combined_csv}")
        
        # Create a summary CSV with max and min strain for each notch and point
        if not summary_data:  # If summary_data wasn't already populated in the loop above
            for notch_name, point_dict in strain_data_dict.items():
                for point_id, df in point_dict.items():
                    # Filter by valid data points
                    valid_df = df[df['Valid'] == True] if 'Valid' in df.columns else df
                    
                    if len(valid_df) > 0 and 'MaxStrain' in valid_df.columns and 'MinStrain' in valid_df.columns:
                        max_strain = valid_df['MaxStrain'].max()
                        min_strain = valid_df['MinStrain'].min()
                        
                        summary_data.append({
                            'Notch': notch_name,
                            'PointID': point_id,
                            'MaxStrainPeak': max_strain,
                            'MinStrainPeak': min_strain,
                            'Subset_ID': point_id,  # Use point_id as the subset_id since they're the same
                            'Valid_Frames': len(valid_df),
                            'Total_Frames': len(df)
                        })
        
        # Create the summary DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        summary_csv = os.path.join(data_dir, f"strain_summary_{param_str}_{timestamp}.csv")
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"Saved strain summary data to {summary_csv}")
        
        return data_dir
    
    except Exception as e:
        logger.error(f"Error saving strain data to CSV: {str(e)}")
        return None

def generate_all_point_maps(notch_folders, settings, logger):
    """
    Generate point maps for all notch folders, allowing multiple points per notch.
    
    Args:
        notch_folders (list): List of notch folder paths.
        settings (dict): Settings dictionary.
        logger (logging.Logger): Logger object.
        
    Returns:
        bool: True if at least one point is selected for any notch, False otherwise.
    """
    # Create pointmaps directory in the data directory
    pointmaps_dir = os.path.join(settings['data_dir'], "pointmaps")
    os.makedirs(pointmaps_dir, exist_ok=True)
    
    # Initialize selected_points dict in settings if it doesn't exist
    if 'selected_points' not in settings:
        settings['selected_points'] = {}
    
    # Process each notch folder
    for notch_folder in notch_folders:
        notch_name = os.path.basename(notch_folder)
        analysis_folder = get_analysis_folder_path(
            notch_folder,
            settings['cycle'],
            settings['subset'],
            settings['step'],
            settings['vsg']
        )
        
        # Skip if analysis folder doesn't exist
        if not os.path.exists(analysis_folder):
            logger.error(f"Analysis folder not found for {notch_name}: {analysis_folder}")
            continue
        
        # Get existing points for this notch if any
        existing_points = []
        if notch_name in settings['selected_points']:
            # Handle both list and single integer cases (for backward compatibility)
            existing_points_data = settings['selected_points'][notch_name]
            if isinstance(existing_points_data, list):
                existing_points = existing_points_data
            else:
                # Convert single value to list
                existing_points = [existing_points_data]
        
        # Generate point map and let user select points
        logger.info(f"Generating point map for {notch_name}...\nUse click to select/deselect points - Multiple selections allowed")
        selected_ids = generate_point_map(notch_folder, analysis_folder, logger, existing_points)
        
        if selected_ids:
            # Save selected points to settings
            settings['selected_points'][notch_name] = selected_ids
            logger.info(f"Selected subset IDs {selected_ids} for {notch_name}")
            
            # Save point map image
            # This is handled in the generate_point_map function
        else:
            logger.warning(f"No points selected for {notch_name}")
    
    # Save updated settings with selected points
    save_settings(settings)
    
    # Check if we have at least one point selected across all notches
    has_selections = False
    for notch_name, points in settings['selected_points'].items():
        if points:  # If there's at least one point selected
            has_selections = True
            break
    
    return has_selections

def analyze_all_notches(notch_folders, settings, logger):
    """
    Process strain data for all notch folders with multiple points per notch.
    
    Args:
        notch_folders (list): List of notch folder paths.
        settings (dict): Settings dictionary.
        logger (logging.Logger): Logger object.
        
    Returns:
        dict: Dictionary with structure {notch_name: {point_id: DataFrame}}.
    """
    # Check if at least some notches have selected points
    selected_points = settings.get('selected_points', {})
    if not selected_points:
        logger.error("No points selected for any notch. Please run again to select points.")
        return None
    
    # Load timestamp data if camera log file is specified and timestamps are enabled
    timestamp_map = None
    temperature_map = None
    frame_index_range = None
    
    if settings.get('use_timestamps', False) and settings.get('camera_log_file'):
        camera_log_file = settings['camera_log_file']
        if os.path.exists(camera_log_file):
            logger.info(f"Loading timestamp data from {camera_log_file}")
            timestamp_map, temperature_map = parse_camera_log(camera_log_file)
            
            if timestamp_map:
                # Extract frame indices from input.xml for the first notch folder (should be the same for all notches in the same cycle)
                if notch_folders:
                    first_notch = notch_folders[0]
                    frame_index_range = extract_frame_indices_from_xml(
                        first_notch,
                        settings['cycle'],
                        settings['subset'],
                        settings['step'],
                        settings['vsg']
                    )
                    
                    if frame_index_range:
                        start_index, end_index = frame_index_range
                        logger.info(f"Using frame index range: {start_index} to {end_index} for timestamps")
                    else:
                        logger.warning("Could not determine frame index range from input.xml")
            else:
                logger.warning(f"No timestamp data could be extracted from {camera_log_file}")
        else:
            logger.warning(f"Camera log file not found: {camera_log_file}")
    
    # Process each notch folder with potentially multiple points per notch
    strain_data_dict = {}
    
    for notch_folder in notch_folders:
        notch_name = os.path.basename(notch_folder)
        
        # Get selected points for this notch
        notch_points = selected_points.get(notch_name, [])
        
        # Handle both list and single integer cases (for backward compatibility)
        if not isinstance(notch_points, list):
            notch_points = [notch_points] if notch_points is not None else []
        
        if not notch_points:
            logger.warning(f"No points selected for {notch_name}, skipping")
            continue
        
        # Get analysis folder
        analysis_folder = get_analysis_folder_path(
            notch_folder,
            settings['cycle'],
            settings['subset'],
            settings['step'],
            settings['vsg']
        )
        
        # Skip if analysis folder doesn't exist
        if not os.path.exists(analysis_folder):
            logger.error(f"Analysis folder not found for {notch_name}: {analysis_folder}")
            continue
        
        # Create a dictionary to store data for each point in this notch
        notch_data = {}
        
        # Process strain data for each selected point
        for point_id in notch_points:
            logger.info(f"Processing strain data for {notch_name}, point ID {point_id}...")
            
            strain_data = process_strain_data(
                notch_folder,
                analysis_folder,
                point_id,
                settings.get('gamma_threshold', GAMMA_THRESHOLD),
                settings.get('sigma_threshold', SIGMA_THRESHOLD),
                logger,
                timestamp_map,
                temperature_map,
                frame_index_range
            )
            
            if strain_data is not None:
                # Store the data for this point
                notch_data[point_id] = strain_data
                logger.info(f"Successfully processed {len(strain_data)} frames for {notch_name}, point ID {point_id}")
            else:
                logger.warning(f"Failed to process strain data for {notch_name}, point ID {point_id}")
        
        # Only add this notch to the result if we have data for at least one point
        if notch_data:
            strain_data_dict[notch_name] = notch_data
    
    return strain_data_dict

def main():
    """
    Main function to run the cycle post-processing pipeline.
    """
    # Set up logging
    logger = setup_logging()
    logger.info("Starting cycle post-processing")
    
    # Check if settings file exists
    if not os.path.exists(DEFAULT_SETTINGS_FILE):
        # Create a new settings file with default values
        settings = load_settings()  # This will create the file since it doesn't exist
        logger.info(f"Created new settings file at {DEFAULT_SETTINGS_FILE}")
        print(f"\nA new settings file has been created at:\n{os.path.abspath(DEFAULT_SETTINGS_FILE)}\n")
        print("Please edit this file to configure your analysis parameters (cycle, subset, step, VSG).")
        print("Then run this script again to continue with the analysis.")
        return  # Exit the script after creating the settings file
    
    # Load settings from existing file
    settings = load_settings()
    logger.info(f"Loaded settings from {DEFAULT_SETTINGS_FILE}")
    
    # Get base output directory from settings
    base_output_dir = settings.get('output_dir', os.path.join(settings['data_dir'], "analysis_results"))
    
    # Add timestamp to output directory to prevent overwriting previous analyses
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_output_dir, f"analysis_{timestamp}")
    logger.info(f"Using timestamped output directory: {output_dir}")
    
    # Find notch folders
    notch_folders = find_notch_folders(settings['data_dir'])
    if not notch_folders:
        logger.error(f"No notch folders found in {settings['data_dir']}")
        print(f"\nERROR: No notch folders found in the data directory.\nPlease check the 'data_dir' setting in {DEFAULT_SETTINGS_FILE}")
        return
    
    logger.info(f"Found {len(notch_folders)} notch folders")
    print(f"Found {len(notch_folders)} notch folders in {settings['data_dir']}")
    
    # Check if the user already has selected points
    selected_points = settings.get('selected_points', {})
    has_valid_selections = False
    
    # Check if the selected points are valid and exist in the notch folders
    if selected_points:
        notch_names = [os.path.basename(folder) for folder in notch_folders]
        for notch_name, points in selected_points.items():
            if notch_name in notch_names and points:  # If the notch exists and has points
                has_valid_selections = True
                break
    
    # If no valid selections exist, prompt the user to select points
    if not has_valid_selections:
        logger.info("No valid point selections found. Need to select points.")
        print("\nYou need to select analysis points for each notch.")
        print("A point map will be displayed for each notch. Click on the point you want to analyze.")
        print("After selecting all points, the script will save your selections and exit.")
        print("Run the script again to perform the full analysis.\n")
        
        # Generate point maps and let user select points
        generate_all_point_maps(notch_folders, settings, logger)
        
        # Save settings with selected points
        save_settings(settings)
        
        logger.info("Point selection complete. Run the script again to analyze data.")
        print("\nPoint selection complete!")
        print(f"Selected points have been saved to {DEFAULT_SETTINGS_FILE}")
        print("Run the script again to analyze the data with these points.")
        return  # Exit after point selection
    
    # If we get here, we have some valid point selections and can proceed with analysis
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved to {output_dir}")
    
    # Process strain data for all notches
    print("\nAnalyzing strain data using selected points...")
    strain_data_dict = analyze_all_notches(notch_folders, settings, logger)
    if not strain_data_dict or len(strain_data_dict) == 0:
        logger.error("No strain data to process")
        print("\nERROR: No valid strain data was found to process.")
        print("Check the analysis parameters in your settings file and make sure the data exists.")
        return
    
    # Create plots if enabled
    if settings.get('create_plots', True):
        logger.info("Creating strain plots...")
        print("Creating strain plots...")
        max_strain_plots, min_strain_plots = create_strain_plots(strain_data_dict, output_dir, settings, logger)
    else:
        logger.info("Plot creation disabled in settings")
    
    # Save data to CSV if enabled
    if settings.get('save_csv', True):
        logger.info("Saving strain data to CSV...")
        print("Saving strain data to CSV...")
        data_dir = save_strain_data_to_csv(strain_data_dict, output_dir, settings, logger)
    else:
        logger.info("CSV saving disabled in settings")
    
    # Save a copy of the settings file to the output directory for reference
    settings_backup = os.path.join(output_dir, f"settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(settings_backup, 'w') as f:
        # Need to convert numpy types for JSON serialization
        clean_settings = {}
        for key, value in settings.items():
            if key == 'selected_points':
                # Handle the selected_points dictionary separately
                clean_selected_points = {}
                for notch, point_ids in value.items():
                    # Handle multiple points per notch (list of points)
                    if isinstance(point_ids, list):
                        clean_points = []
                        for pid in point_ids:
                            # Convert numpy int types to Python int
                            if hasattr(pid, 'item') and callable(getattr(pid, 'item')):
                                clean_points.append(pid.item())
                            else:
                                clean_points.append(pid)
                        clean_selected_points[notch] = clean_points
                    elif point_ids is not None:
                        # For backward compatibility with single point selection
                        if hasattr(point_ids, 'item') and callable(getattr(point_ids, 'item')):
                            clean_selected_points[notch] = point_ids.item()
                        else:
                            clean_selected_points[notch] = point_ids
                    else:
                        clean_selected_points[notch] = None
                clean_settings[key] = clean_selected_points
            else:
                # Handle other settings values
                if hasattr(value, 'item') and callable(getattr(value, 'item')):  # For numpy types
                    clean_settings[key] = value.item()
                else:
                    clean_settings[key] = value
        json.dump(clean_settings, f, indent=4)
    
    logger.info("Cycle post-processing complete")
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Settings saved to {settings_backup}")
    
    print(f"\nProcessing complete! Results saved to:\n{output_dir}")
    if settings.get('create_plots', True):
        print("\nPlots created:")
        print(f"- Maximum strain plots for each notch")
        print(f"- Minimum strain plots for each notch")
        print(f"- Combined plots for all notches")
    
    if settings.get('save_csv', True):
        print("\nCSV files created:")
        print(f"- Individual strain data files for each notch")
        print(f"- Combined strain data for all notches")
        print(f"- Summary of strain extremes for each notch")

if __name__ == "__main__":
    main()
