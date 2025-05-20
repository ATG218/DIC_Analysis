import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import glob
import os
import re
from multiprocessing import Pool, cpu_count

def process_folder(folder_path):
    """Process a single DICe folder and return all strain data."""
    print(f"\nProcessing folder: {folder_path}")
    
    # Get all txt files in the results directory
    txt_files = sorted(glob.glob(os.path.join(folder_path, 'results', 'DICe_solution_*.txt')))
    
    if not txt_files:
        print("  No txt files found in results directory")
        return None
    
    print(f"  Found {len(txt_files)} txt files to process")
    
    # Collect all strain data from the time series
    all_exx = []
    all_eyy = []
    all_exy = []
    
    for txt_file in txt_files:
        try:
            # Skip the first few lines of metadata
            df = pd.read_csv(txt_file, skiprows=20)
            
            all_exx.extend(df['VSG_STRAIN_XX'].values)
            all_eyy.extend(df['VSG_STRAIN_YY'].values)
            all_exy.extend(df['VSG_STRAIN_XY'].values)
            
        except Exception as e:
            print(f"    Error processing {os.path.basename(txt_file)}: {str(e)}")
            continue
    
    if not all_exx:
        print("  No data was successfully processed from this folder")
        return None
        
    print(f"  Successfully processed {len(all_exx)} total data points")
    
    # Get notch from folder name
    notch_match = re.search(r'n(\d+)', os.path.basename(folder_path))
    notch = int(notch_match.group(1)) if notch_match else 0
    
    return {
        'folder_name': os.path.basename(folder_path),
        'notch': notch,
        'Exx': np.array(all_exx),
        'Eyy': np.array(all_eyy),
        'Exy': np.array(all_exy)
    }

def analyze_strain_distribution(base_directory):
    """Analyze strain distributions across all DICe parameter folders."""
    # Find all DICe parameter folders (e.g., DICe_subset27_step02_VSG004)
    dice_folders = sorted([
        os.path.join(base_directory, name)
        for name in os.listdir(base_directory)
        if os.path.isdir(os.path.join(base_directory, name)) and name.startswith('DICe_subset')
    ])

    print(f"Found {len(dice_folders)} DICe parameter folders to process")

    # Process folders concurrently
    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        results = pool.map(process_folder, dice_folders)

    # Dictionary to store all strain data by folder
    folder_data = {}

    # Organize results
    for result in results:
        if result:
            folder_name = result.pop('folder_name')
            folder_data[folder_name] = result

    print(f"\nSuccessfully processed {len(folder_data)} folders")
    if len(folder_data) == 0:
        print("No data was processed successfully. Check folder pattern and file contents.")
        return None
    
    return folder_data

def plot_strain_histograms(folder_data, output_dir=None):
    """Plot histograms and fit Gaussians for strain data, in groups of 10 folders.
    Also: (1) fit and report stddev omitting zeros, (2) scatter plot nonzero-value positions."""
    if not folder_data:
        print("No data to plot")
        return
    
    plt.style.use('default')
    strain_components = ['Exx', 'Eyy', 'Exy']
    titles = ['XX Strain', 'YY Strain', 'XY Strain']

    # Calculate global statistics (including zeros)
    all_data = {comp: np.concatenate([data[comp] for data in folder_data.values()])
                for comp in strain_components}
    global_stats = {}
    for strain in strain_components:
        data = all_data[strain]
        global_stats[strain] = {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.mean(data) - 4*np.std(data),
            'max': np.mean(data) + 4*np.std(data)
        }

    # Sort folders for consistent groupings
    sorted_items = sorted(folder_data.items(), key=lambda x: x[0])
    n_folders = len(sorted_items)
    n_groups = (n_folders + 9) // 10

    for group_idx in range(n_groups):
        group_items = sorted_items[group_idx*10:(group_idx+1)*10]
        if not group_items:
            continue
        
        fig = plt.figure(figsize=(15, 25))
        gs = plt.GridSpec(3, 2, figure=fig, hspace=0.4, top=0.95, bottom=0.05, left=0.1, right=0.95)
        axes_hist = [fig.add_subplot(gs[i, 0]) for i in range(3)]
        axes_qq = [fig.add_subplot(gs[i, 1]) for i in range(3)]
        fig.suptitle(f'Strain Distribution Analysis (Folders {group_idx*10+1}–{min((group_idx+1)*10, n_folders)})', fontsize=14, y=0.98)
        colors = plt.cm.tab10(np.linspace(0, 1, len(group_items)))

        for i, (folder_name, data) in enumerate(group_items):
            # Extract subset, step, vsg from folder name
            match = re.search(r'subset(\d+)_step(\d+)_VSG(\d+)', folder_name)
            if match:
                subset, step, vsg = match.groups()
            else:
                subset = step = vsg = '?'
            legend_info = f"subt={subset} stp={step} vsg={vsg} (n={len(data['Exx'])})"

            for j, (strain, title) in enumerate(zip(strain_components, titles)):
                strain_data = data[strain]
                # Omit all zero values for new fit
                nonzero_strain = strain_data[strain_data != 0]
                mu, std = norm.fit(strain_data)
                mu_nz, std_nz = (np.nan, np.nan)
                if len(nonzero_strain) > 0:
                    mu_nz, std_nz = norm.fit(nonzero_strain)
                bins = np.linspace(global_stats[strain]['min'], global_stats[strain]['max'], 50)
                counts, bins, _ = axes_hist[j].hist(strain_data, bins=bins, alpha=0.3, color=colors[i], label=legend_info)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                bin_width = bins[1] - bins[0]
                scale = len(strain_data) * bin_width
                gaussian = scale * norm.pdf(bin_centers, mu, std)
                axes_hist[j].plot(bin_centers, gaussian, '--', color=colors[i], linewidth=1)
                axes_hist[j].grid(True, alpha=0.3)
                axes_hist[j].set_title(f'{title} Distribution', fontsize=12, pad=20)
                axes_hist[j].set_xlabel('Strain', fontsize=10)
                axes_hist[j].set_ylabel('Count', fontsize=10)
                axes_hist[j].set_xlim(global_stats[strain]['min'], global_stats[strain]['max'])
                # Add legend to first subplot only
                if j == 0:
                    axes_hist[j].legend(fontsize=8, loc='upper right')
                # Add global stats and stddevs
                stats_text = (f'Global Stats:\n'
                             f'Mean: {global_stats[strain]["mean"]:.6f}\n'
                             f'Std (all): {global_stats[strain]["std"]:.6f}\n'
                             f'Std (no 0): {std_nz:.6f}\n'
                             f'n = {len(strain_data)}\n\n'
                             f'Dotted lines show\nGaussian fit')
                axes_hist[j].text(0.02, 0.98, stats_text,
                                transform=axes_hist[j].transAxes,
                                verticalalignment='top',
                                fontsize=8,
                                bbox=dict(facecolor='white', alpha=0.8))
                # Create QQ plot
                from scipy.stats import probplot
                probplot(strain_data, dist="norm", plot=axes_qq[j])
                axes_qq[j].set_title(f'{title} Q-Q Plot\n(Points on line = perfectly normal)', fontsize=12, pad=20)
                axes_qq[j].grid(True, alpha=0.3)

        if output_dir:
            plt.savefig(os.path.join(output_dir, f'histogram_analysis_group{group_idx+1}.png'), dpi=300)
        else:
            plt.show()

    # --- Scatter plot of nonzero positions for each folder (aggregate ALL txt files) ---
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for folder_name, data in folder_data.items():
        # Find a representative results folder
        folder_path = None
        for base in [os.getcwd(), os.path.dirname(__file__)]:
            candidate = os.path.join(base, folder_name, 'results')
            if os.path.exists(candidate):
                folder_path = candidate
                break
        if folder_path is None:
            print(f"[DEBUG] Results folder not found for {folder_name}")
            continue
        # Find all results txt files
        txt_files = sorted(glob.glob(os.path.join(folder_path, 'DICe_solution_*.txt')))
        if not txt_files:
            print(f"[DEBUG] No txt files found in {folder_path}")
            continue
        # For each strain, aggregate all nonzero positions from all files
        for strain in strain_components:
            x_all = []
            y_all = []
            for txt_file in txt_files:
                try:
                    df = pd.read_csv(txt_file, skiprows=20)
                except Exception as e:
                    print(f"[DEBUG] Failed to read {txt_file}: {e}")
                    continue
                # Try to get X/Y columns
                xcol, ycol = None, None
                for candx in ['COORDINATE_X', 'X', 'x']:
                    if candx in df.columns:
                        xcol = candx
                        break
                for candy in ['COORDINATE_Y', 'Y', 'y']:
                    if candy in df.columns:
                        ycol = candy
                        break
                if xcol is None or ycol is None or strain not in df.columns:
                    print(f"[DEBUG] Missing columns in {txt_file}: xcol={xcol}, ycol={ycol}, strain={strain in df.columns}")
                    continue
                nonzero = df[strain] != 0
                xvals = df.loc[nonzero, xcol].values
                yvals = df.loc[nonzero, ycol].values
                x_all.append(xvals)
                y_all.append(yvals)
            # Combine all points
            if x_all and y_all and any(len(x) > 0 for x in x_all):
                xcat = np.concatenate(x_all)
                ycat = np.concatenate(y_all)
                print(f"[DEBUG] {folder_name} {strain}: {len(xcat)} nonzero points, saving scatter plot.")
                plt.figure(figsize=(6, 5))
                plt.scatter(xcat, ycat, s=2, alpha=0.5)
                plt.title(f"Nonzero {strain} positions in {folder_name} (all files)")
                plt.xlabel(xcol)
                plt.ylabel(ycol)
                plt.tight_layout()
                if output_dir:
                    plt.savefig(os.path.join(output_dir, f'scatter_{folder_name}_{strain}.png'), dpi=200)
                else:
                    plt.show()
            else:
                print(f"[DEBUG] {folder_name} {strain}: No nonzero points found, skipping scatter plot.")

    # Print detailed statistics (fixed: compute mean/std directly from data arrays)
    print("\nDetailed Strain Analysis:")
    print("=======================")
    print(f"{'Configuration':^25} | {'Mean':^12} | {'Std Dev':^12} | {'Bias from Global':^15}")
    print("-" * 70)
    sorted_folders = sorted(folder_data.items(), key=lambda x: folder_data[x[0]]['notch'])
    for folder_name, data in sorted_folders:
        print(f"Notch {data['notch']}:")
        for strain in strain_components:
            arr = data[strain]
            folder_mean = np.mean(arr)
            folder_std = np.std(arr)
            global_mean = global_stats[strain]['mean']
            bias = folder_mean - global_mean
            print(f"{strain + ':':^25} | {folder_mean:^12.6f} | {folder_std:^12.6f} | {bias:^+15.6f}")

if __name__ == "__main__":
    # Example usage for new directory structure
    base_dir = r"C:\Users\alext\OneDrive\Documents\UROP25\data\OneDrive_2025-05-09\noisefloorbmp_sorted\Notch_01\DICe"
    folder_data = analyze_strain_distribution(base_dir)
    plot_strain_histograms(folder_data, r"C:\Users\alext\OneDrive\Documents\UROP25\Analysis_Pipeline\test")
