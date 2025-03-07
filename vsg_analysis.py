import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re

# Set the directory containing your CSV files
data_directory = r"C:\Users\alext\OneDrive\Documents\UROP25\DiceCLI\DICe_Processing\parameterStudy\noiseParametric8\n8\step2\strain_analysis"

# Change to the specified directory
os.chdir(data_directory)

# Get all CSV files in the directory
csv_files = glob.glob('*.csv')

# Dictionary to store data for each subset size
subset_data = {}

# Regular expression pattern to extract VSG size and subset size
pattern = r'subset6739_(\d+)_(\d+)_'

# Process each CSV file
for csv_file in csv_files:
    # Extract VSG size and subset size from filename
    match = re.search(pattern, csv_file)
    if match:
        vsg_size = int(match.group(1))
        subset_size = int(match.group(2))
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Calculate standard deviations for each strain component
        std_exx = df['Exx'].std()
        std_eyy = df['Eyy'].std()
        std_exy = df['Exy'].std()
        
        # Initialize data structure for this subset size if it doesn't exist
        if subset_size not in subset_data:
            subset_data[subset_size] = {
                'vsg_sizes': [],
                'std_exx': [],
                'std_eyy': [],
                'std_exy': []
            }
        
        # Store the data
        subset_data[subset_size]['vsg_sizes'].append(vsg_size)
        subset_data[subset_size]['std_exx'].append(std_exx)
        subset_data[subset_size]['std_eyy'].append(std_eyy)
        subset_data[subset_size]['std_exy'].append(std_exy)

# Create three separate plots for Exx, Eyy, and Exy
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Strain Standard Deviation vs VSG Size for Different Subset Sizes (Step Size 2)')

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
    ax.set_ylabel('Standard Deviation')
    ax.set_title(f'{title} Strain Noise')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis to show actual numbers
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.xaxis.set_minor_formatter(plt.ScalarFormatter())

# Only show legend on the last plot (Exy)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# Print numerical results
print("\nNumerical Results:")
print("=================")
for subset_size, data in sorted(subset_data.items()):
    print(f"\nSubset Size: {subset_size}")
    for i, vsg in enumerate(sorted(data['vsg_sizes'])):
        idx = np.argsort(data['vsg_sizes'])[i]
        print(f"VSG Size: {vsg}")
        print(f"  Exx std: {data['std_exx'][idx]:.6f}")
        print(f"  Eyy std: {data['std_eyy'][idx]:.6f}")
        print(f"  Exy std: {data['std_exy'][idx]:.6f}")
