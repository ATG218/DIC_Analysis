import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import glob
import os

# Set the directory containing your CSV files here
data_directory = r"c:\Users\alext\OneDrive\Documents\UROP25\DiceCLI\DICe_Processing\noiseStudy\strain_analysis"

# Change to the specified directory
os.chdir(data_directory)

# Get all CSV files in the directory
csv_files = glob.glob('*.csv')

# Create figure with subplots for each strain component
plt.style.use('default')
fig, axes = plt.subplots(3, 1, figsize=(12, 15))
fig.suptitle('Strain Noise Distribution Analysis Across Notches', fontsize=14, y=0.95)

strain_components = ['Exx', 'Eyy', 'Exy']
titles = ['XX Strain', 'YY Strain', 'XY Strain']
colors = plt.cm.tab10(np.linspace(0, 1, len(csv_files)))

# First pass to get overall statistics
all_data = {comp: [] for comp in strain_components}
for csv_file in sorted(csv_files):
    df = pd.read_csv(csv_file)
    for strain in strain_components:
        all_data[strain].extend(df[strain].values)

# Calculate global statistics and limits
global_stats = {}
for strain in strain_components:
    data = np.array(all_data[strain])
    global_stats[strain] = {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.mean(data) - 4*np.std(data),
        'max': np.mean(data) + 4*np.std(data)
    }

# Store notch statistics
notch_stats = {}

# Plot histograms and fit Gaussians
for i, csv_file in enumerate(sorted(csv_files)):
    notch_num = csv_file.split('_')[1]
    df = pd.read_csv(csv_file)
    
    notch_stats[notch_num] = {}
    
    for j, (strain, title) in enumerate(zip(strain_components, titles)):
        data = df[strain]
        mu, std = norm.fit(data)
        notch_stats[notch_num][strain] = {'mean': mu, 'std': std}
        
        bins = np.linspace(global_stats[strain]['min'], 
                         global_stats[strain]['max'], 200)
        
        # Plot histogram
        axes[j].hist(data, bins=bins, density=True, alpha=0.15,
                    color=colors[i], label=f'Notch {notch_num}')
        
        # Fit and plot Gaussian (without label)
        x = np.linspace(global_stats[strain]['min'], 
                       global_stats[strain]['max'], 1000)
        p = norm.pdf(x, mu, std)
        axes[j].plot(x, p, color=colors[i], linewidth=2, linestyle='--')
        
        # Add vertical line for mean
        axes[j].axvline(x=mu, color=colors[i], alpha=0.7, linestyle='-', 
                       label=f'Mean {notch_num}')
        
        # Add grid and set labels
        axes[j].grid(True, alpha=0.3)
        axes[j].set_title(title, fontsize=12, pad=10)
        axes[j].set_xlabel('Strain', fontsize=10)
        axes[j].set_ylabel('Density', fontsize=10)
        axes[j].set_xlim(global_stats[strain]['min'], global_stats[strain]['max'])
        
        # Add legend to first subplot only
        if j == 0:
            axes[j].legend(fontsize=8, loc='upper right')
            
        # Add global stats to each plot
        stats_text = (f'Global Stats:\n'
                     f'Mean: {global_stats[strain]["mean"]:.6f}\n'
                     f'Std: {global_stats[strain]["std"]:.6f}\n'
                     f'\nVertical lines show mean value\n'
                     f'for each notch')
        axes[j].text(0.02, 0.98, stats_text,
                    transform=axes[j].transAxes,
                    verticalalignment='top',
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()

# Print detailed statistics for each notch
print("\nDetailed Noise Analysis:")
print("======================")
for notch_num in sorted(notch_stats.keys()):
    print(f"\nNotch {notch_num}:")
    for strain in strain_components:
        stats = notch_stats[notch_num][strain]
        bias = stats['mean'] - global_stats[strain]['mean']
        bias_direction = "higher than" if bias > 0 else "lower than" if bias < 0 else "at"
        print(f"{strain}:")
        print(f"  Mean: {stats['mean']:.6f} ({bias_direction} global mean by {abs(bias):.6f})")
        print(f"  Std:  {stats['std']:.6f}")

plt.show()
