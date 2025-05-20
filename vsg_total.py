import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
from multiprocessing import Pool, cpu_count

def process_folder(folder_path):
    print(f"\nProcessing folder: {folder_path}")
    # Regular expression to extract VSG size and subset size from folder name
    folder_pattern = r'DICe_subset(\d+)_step(\d+)_VSG(\d+)'
    match = re.search(folder_pattern, os.path.basename(folder_path))
    
    if not match:
        print("  No match found for folder pattern")
        return None
    
    subset_size = int(match.group(1))
    step_size = int(match.group(2))
    vsg_size = int(match.group(3))
    print(f"  Found: Subset Size = {subset_size}, Step Size = {step_size}, VSG Size = {vsg_size}")
    
    # Get all txt files in the results directory
    txt_files = sorted(glob.glob(os.path.join(folder_path, 'results', 'DICe_solution_*.txt')))
    
    if not txt_files:
        print("  No txt files found in results directory")
        return None
    
    # Process each txt file in the time series
    all_std_exx = []
    all_std_eyy = []
    all_std_exy = []
    
    for txt_file in txt_files:
        try:
            # Skip the first few lines of metadata
            df = pd.read_csv(txt_file, skiprows=20)
            
            # Calculate standard deviations for each strain component
            std_exx = df['VSG_STRAIN_XX'].std()
            std_eyy = df['VSG_STRAIN_YY'].std()
            std_exy = df['VSG_STRAIN_XY'].std()
            
            all_std_exx.append(std_exx)
            all_std_eyy.append(std_eyy)
            all_std_exy.append(std_exy)
        except Exception as e:
            print(f"    Error processing {os.path.basename(txt_file)}: {str(e)}")
            continue
    
    # Calculate mean standard deviation across the time series
    return {
        'subset_size': subset_size,
        'step_size': step_size,
        'vsg_size': vsg_size,
        'mean_std_exx': np.mean(all_std_exx),
        'mean_std_eyy': np.mean(all_std_eyy),
        'mean_std_exy': np.mean(all_std_exy)
    }

def analyze_vsg_data(base_directory):
    # Dictionary to store data for each subset size
    subset_data = {}
    step_size = None
    
    # Get all DICe folders
    dice_folders = glob.glob(os.path.join(base_directory, 'DICe_subset*_step*_VSG*'))
    
    # Process folders concurrently
    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        results = pool.map(process_folder, dice_folders)
    
    # Organize results
    for result in results:
        if result:
            subset_size = result['subset_size']
            if step_size is None:
                step_size = result['step_size']
            
            if subset_size not in subset_data:
                subset_data[subset_size] = {
                    'vsg_sizes': [],
                    'std_exx': [],
                    'std_eyy': [],
                    'std_exy': []
                }
            
            subset_data[subset_size]['vsg_sizes'].append(result['vsg_size'])
            subset_data[subset_size]['std_exx'].append(result['mean_std_exx'])
            subset_data[subset_size]['std_eyy'].append(result['mean_std_eyy'])
            subset_data[subset_size]['std_exy'].append(result['mean_std_exy'])
    
    return subset_data, step_size

def plot_vsg_analysis(subset_data, step_size, output_dir=None):
    # Create three separate plots for Exx, Eyy, and Exy
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Mean Strain Standard Deviation vs VSG Size for Different Subset Sizes (Notch 1)')
    
    # Colors for different subset sizes
    colors = plt.cm.viridis(np.linspace(0, 1, len(subset_data)))
    
    # Plot for each strain component
    for i, (subset_size, data) in enumerate(sorted(subset_data.items())):
        # Sort data by VSG size
        sort_idx = np.argsort(data['vsg_sizes'])
        vsg_sizes = np.array(data['vsg_sizes'])[sort_idx]
        
        # Exx plot
        std_exx = np.array(data['std_exx'])[sort_idx]
        ax1.plot(vsg_sizes, std_exx, 'o-', color=colors[i], 
                 label=f'Subset Size {subset_size}')
        
        # Eyy plot
        std_eyy = np.array(data['std_eyy'])[sort_idx]
        ax2.plot(vsg_sizes, std_eyy, 'o-', color=colors[i], 
                 label=f'Subset Size {subset_size}')
        
        # Exy plot
        std_exy = np.array(data['std_exy'])[sort_idx]
        ax3.plot(vsg_sizes, std_exy, 'o-', color=colors[i], 
                 label=f'Subset Size {subset_size}')
    
    # Configure plots
    for ax, title in zip([ax1, ax2, ax3], ['Exx', 'Eyy', 'Exy']):
        ax.set_xlabel('VSG Size')
        ax.set_ylabel('Mean Standard Deviation')
        ax.set_title(f'{title} Strain Noise')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_minor_formatter(plt.ScalarFormatter())
    
    # Only show legend on the last plot (Exy)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'vsg_analysis.png'), bbox_inches='tight', dpi=300)
    else:
        plt.show()

def print_statistics_table(subset_data):
    # Print table header
    print("\nStatistical Analysis Results")
    print("=" * 80)
    print(f"{'Subset Size':^12} | {'VSG Size':^10} | {'Exx Std':^15} | {'Eyy Std':^15} | {'Exy Std':^15}")
    print("-" * 80)
    
    # Print data rows
    for subset_size, data in sorted(subset_data.items()):
        first_row = True
        for i, vsg in enumerate(sorted(data['vsg_sizes'])):
            idx = np.argsort(data['vsg_sizes'])[i]
            if first_row:
                subset_str = f"{subset_size:^12}"
                first_row = False
            else:
                subset_str = " " * 12
            
            print(f"{subset_str} | {vsg:^10} | {data['std_exx'][idx]:^15.6f} | {data['std_eyy'][idx]:^15.6f} | {data['std_exy'][idx]:^15.6f}")
        print("-" * 80)

if __name__ == "__main__":
    # Example usage
    base_dir = r"C:\Users\alext\OneDrive\Documents\UROP25\data\OneDrive_2025-05-09\noisefloorbmp_sorted\Notch_01\DICe"
    subset_data, step_size = analyze_vsg_data(base_dir)
    plot_vsg_analysis(subset_data, step_size)
    print_statistics_table(subset_data)
