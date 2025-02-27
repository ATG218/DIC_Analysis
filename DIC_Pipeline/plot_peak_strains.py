import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict

def extract_info_from_filename(filename: str) -> Dict:
    """Extract information from the filename."""
    # Example filename: n7_subset695_30_21_principal_data.csv
    parts = os.path.basename(filename).split('_')
    if len(parts) >= 4:
        try:
            return {
                'notch': parts[0],  # Take the full first field as the notch identifier
                'subset_id': int(parts[1].replace('subset', '')),
                'strain_gauge': int(parts[2]),
                'subset_size': int(parts[3])
            }
        except (ValueError, IndexError):
            return None
    return None

def process_csv_files(directories: List[str], use_average: bool = False, x_axis_type: str = 'subset') -> Dict:
    """Process all CSV files in the given directories."""
    data = {'e1': {}, 'e2': {}}
    
    for directory in directories:
        try:
            csv_files = glob.glob(os.path.join(directory, '*.csv'))
            if not csv_files:
                print(f"\nWarning: No CSV files found in {directory}")
                continue
                
            print(f"\nProcessing directory: {directory}")
            print(f"Found {len(csv_files)} CSV files")
            
            # Dictionary to store temporary data for averaging duplicate x values
            temp_data = {}
            
            for csv_file in csv_files:
                try:
                    info = extract_info_from_filename(csv_file)
                    if not info:
                        print(f"Warning: Skipping file {os.path.basename(csv_file)} - couldn't parse filename")
                        continue
                    
                    # Read the CSV file
                    df = pd.read_csv(csv_file)
                    if 'e1' not in df.columns or 'e2' not in df.columns:
                        print(f"Warning: File {os.path.basename(csv_file)} missing required columns")
                        continue
                    
                    # Calculate either peak or average strains
                    if use_average:
                        strain_e1 = df['e1'].mean()
                        strain_e2 = df['e2'].mean()
                    else:
                        strain_e1 = df['e1'].max()
                        strain_e2 = df['e2'].min()  # Using min since e2 is typically negative
                    
                    # Store data organized by notch identifier
                    notch_key = directory.split("\\")[-2]
                    
                    # Choose x-axis value based on type
                    x_value = info['subset_size'] if x_axis_type == 'subset' else info['strain_gauge']
                    
                    # Store in temporary dictionary to handle duplicates
                    if notch_key not in temp_data:
                        temp_data[notch_key] = {}
                    if x_value not in temp_data[notch_key]:
                        temp_data[notch_key][x_value] = {'e1': [], 'e2': []}
                    
                    temp_data[notch_key][x_value]['e1'].append(strain_e1)
                    temp_data[notch_key][x_value]['e2'].append(strain_e2)
                    
                    print(f"Processed: {os.path.basename(csv_file)}")
                    print(f"  X: {x_value}, e1: {strain_e1:.6f}, e2: {strain_e2:.6f}")
                except Exception as e:
                    print(f"Error processing file {os.path.basename(csv_file)}: {str(e)}")
            
            # Process temporary data and sort by x value
            for notch_key, notch_data in temp_data.items():
                if notch_key not in data['e1']:
                    data['e1'][notch_key] = {'x_values': [], 'peaks': []}
                    data['e2'][notch_key] = {'x_values': [], 'peaks': []}
                
                # Convert to sorted lists
                sorted_x = sorted(notch_data.keys())
                for x in sorted_x:
                    # Average values if there are duplicates
                    avg_e1 = sum(notch_data[x]['e1']) / len(notch_data[x]['e1'])
                    avg_e2 = sum(notch_data[x]['e2']) / len(notch_data[x]['e2'])
                    
                    data['e1'][notch_key]['x_values'].append(x)
                    data['e1'][notch_key]['peaks'].append(avg_e1)
                    data['e2'][notch_key]['x_values'].append(x)
                    data['e2'][notch_key]['peaks'].append(avg_e2)
                
                print(f"\nSummary for {notch_key}:")
                print("X values:", ", ".join(f"{x}" for x in data['e1'][notch_key]['x_values']))
                print("e1 peaks:", ", ".join(f"{x:.6f}" for x in data['e1'][notch_key]['peaks']))
                print("e2 peaks:", ", ".join(f"{x:.6f}" for x in data['e2'][notch_key]['peaks']))
        except Exception as e:
            print(f"Error processing directory {directory}: {str(e)}")
    
    return data

def plot_peak_strains(data: Dict, use_average: bool = False, x_axis_type: str = 'subset'):
    """Create plots for strains."""
    # Set up the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Color map for different notches
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    strain_type = "Average" if use_average else "Peak"
    x_label = "Subset Size" if x_axis_type == 'subset' else "Strain Gauge Size"
    
    # Plot e1
    for i, (notch, values) in enumerate(data['e1'].items()):
        color = colors[i % len(colors)]
        # Ensure data is sorted by x values
        x_vals = values['x_values']
        peaks = values['peaks']
        
        ax1.plot(x_vals, peaks, 'o-', label=f'{notch}', color=color, markersize=8)
    
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(f'{strain_type} e1 Strain')
    ax1.set_title(f'{strain_type} Maximum Principal Strain vs {x_label}')
    ax1.grid(True)
    ax1.legend()
    
    # Plot e2
    for i, (notch, values) in enumerate(data['e2'].items()):
        color = colors[i % len(colors)]
        # Ensure data is sorted by x values
        x_vals = values['x_values']
        peaks = values['peaks']
        
        ax2.plot(x_vals, peaks, 'o-', label=f'{notch}', color=color, markersize=8)
    
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(f'{strain_type} e2 Strain')
    ax2.set_title(f'{strain_type} Minimum Principal Strain vs {x_label}')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    directories = [
        r"c:\Users\alext\OneDrive\Documents\UROP25\DiceCLI\DICe_Processing\parameterStudy\guageStudy\n5\strain_analysis",
        r"c:\Users\alext\OneDrive\Documents\UROP25\DiceCLI\DICe_Processing\parameterStudy\guageStudy\n6\strain_analysis",
        r"c:\Users\alext\OneDrive\Documents\UROP25\DiceCLI\DICe_Processing\parameterStudy\guageStudy\n7\strain_analysis",
        r"c:\Users\alext\OneDrive\Documents\UROP25\DiceCLI\DICe_Processing\parameterStudy\guageStudy\n8\strain_analysis"
    ]
    
    # Toggle settings
    use_average = True  # Set to True to use average strains instead of peak
    x_axis_type = 'gauge'  # Set to 'subset' for subset size or 'gauge' for strain gauge size
    
    # Process the CSV files
    data = process_csv_files(directories, use_average, x_axis_type)
    
    # Create the plots
    plot_peak_strains(data, use_average, x_axis_type)

if __name__ == "__main__":
    main()
