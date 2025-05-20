import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def extract_subset_size(folder_name):
    match = re.search(r'DICe_subset(\d+)_', folder_name)
    if match:
        return int(match.group(1))
    return None

def count_good_bad_subsets(results_folder, filter_gamma_sigma=False, gamma_max=0.03, sigma_max=0.05):
    import pandas as pd
    dfs = []
    for file in os.listdir(results_folder):
        if file.endswith('.txt') and file.startswith('DICe_solution_'):
            file_path = os.path.join(results_folder, file)
            # Read after header
            with open(file_path, 'r') as f:
                lines = f.readlines()
            # Find header
            header_idx = None
            for idx, line in enumerate(lines):
                if line.startswith('SUBSET_ID'):
                    header_idx = idx
                    break
            if header_idx is not None:
                df = pd.read_csv(file_path, skiprows=header_idx, header=0)
                dfs.append(df)
    if not dfs:
        return 0, 0
    df = pd.concat(dfs, ignore_index=True)
    if filter_gamma_sigma:
        good_mask = (
            (df["STATUS_FLAG"] == 4) &
            (df["GAMMA"] < gamma_max) &
            (df["SIGMA"] < sigma_max)
        )
        good = good_mask.sum()
        bad = (~good_mask).sum()
    else:
        good_mask = (df["STATUS_FLAG"] == 4)
        good = good_mask.sum()
        bad = (~good_mask).sum()
    return good, bad

def main(base_dir):
    # --- User settings ---
    filter_gamma_sigma = True  # Set to True to filter by gamma/sigma, False to ignore
    gamma_max = 0.03           # Adjust as needed
    sigma_max = 0.05           # Adjust as needed
    # ---------------------

    errors_by_subset = {}
    counts_by_subset = {}
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        subset_size = extract_subset_size(folder)
        if subset_size is None:
            continue
        results_folder = os.path.join(folder_path, 'results')
        if not os.path.isdir(results_folder):
            continue
        good, bad = count_good_bad_subsets(results_folder, filter_gamma_sigma=filter_gamma_sigma, gamma_max=gamma_max, sigma_max=sigma_max)
        print(f"Directory: {folder} | Subset Size: {subset_size} | Errors (bad subsets): {bad}")
        if subset_size not in errors_by_subset:
            errors_by_subset[subset_size] = 0
            counts_by_subset[subset_size] = 0
        errors_by_subset[subset_size] += bad
        counts_by_subset[subset_size] += 1
    # Compute average errors per subset size
    subset_sizes = sorted(errors_by_subset.keys())
    avg_bad_counts = [errors_by_subset[size] / counts_by_subset[size] for size in subset_sizes]
    plt.figure(figsize=(10,6))
    plt.plot(subset_sizes, avg_bad_counts, 'ro-', label='Average Bad Subsets (STATUS_FLAG!=4, GAMMA<{}, SIGMA<{})'.format(gamma_max, sigma_max))
    plt.xlabel('Subset Size')
    plt.ylabel('Average Number of Bad Subsets')
    plt.title('Average Number of Bad Subsets vs Subset Size')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Change this path as needed
    BASE_DIR = r'C:\Users\alext\OneDrive\Documents\UROP25\data\OneDrive_2025-05-09\noisefloorbmp_sorted\Notch_01\DICe2'
    main(BASE_DIR)
