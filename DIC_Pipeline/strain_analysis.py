import os
import glob
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple
from pathlib import Path

def read_solution_file(solution_file):
    """Read a DICe solution file and return the data as a DataFrame."""
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

def get_valid_id(first_solution_file):
    """Ask user for a valid subset ID and validate it."""
    df = read_solution_file(first_solution_file)
    valid_ids = set(df['SUBSET_ID'].astype(int))
    
    while True:
        id_input = input("\nEnter a subset ID to analyze (press Enter to exit): ").strip()
        
        if not id_input:  # Empty input
            return None
        
        try:
            subset_id = int(id_input)
            if subset_id in valid_ids:
                return subset_id
            else:
                print(f"Error: ID {subset_id} not found. Valid IDs are: {sorted(valid_ids)}")
        except ValueError:
            print("Error: Please enter a valid integer ID")

def calculate_principal_strains(strain_xx: np.ndarray, strain_yy: np.ndarray, shear: np.ndarray) -> Tuple[float, float]:
    """Calculate principal strains from strain components."""
    # Calculate mean values
    exx = np.mean(strain_xx)
    eyy = np.mean(strain_yy)
    gamma = np.mean(shear)
    
    # Calculate principal strains
    avg = (exx + eyy) / 2
    diff = (exx - eyy) / 2
    radius = np.sqrt(diff * diff + (gamma / 2) * (gamma / 2))
    
    e1 = avg + radius  # Maximum principal strain
    e2 = avg - radius  # Minimum principal strain
    
    return e1, e2

def find_first_solution_file(results_dir):
    """Find the first DICe solution file in the results directory."""
    # Use glob to find all solution files and sort them
    solution_pattern = os.path.join(results_dir, "results", "DICe_solution_*.txt")
    solution_files = sorted(glob.glob(solution_pattern))
    return solution_files[0] if solution_files else None

def process_all_solutions(results_dir, subset_id):
    """Process all solution files and extract strain data for the given subset ID."""
    # Get all solution files
    solution_files = sorted(glob.glob(os.path.join(results_dir, "results", "DICe_solution_*.txt")))
    
    if not solution_files:
        print("No solution files found!")
        return None
    
    # Initialize lists to store data
    times = []
    strain_xx_values = []
    strain_yy_values = []
    shear_values = []
    e1_values = []
    e2_values = []
    
    # Process each solution file
    for i, solution_file in enumerate(solution_files):
        df = read_solution_file(solution_file)
        
        # Find the row for our subset ID
        row = df[df['SUBSET_ID'] == subset_id].iloc[0]
        
        # Extract strain components
        strain_xx = row['VSG_STRAIN_XX']
        strain_yy = row['VSG_STRAIN_YY']
        shear = row['VSG_STRAIN_XY']
        
        # Calculate principal strains
        e1, e2 = calculate_principal_strains(np.array([strain_xx]), np.array([strain_yy]), np.array([shear]))
        
        # Store results
        times.append(i)  # Using frame number as time for now
        strain_xx_values.append(strain_xx)
        strain_yy_values.append(strain_yy)
        shear_values.append(shear)
        e1_values.append(e1)
        e2_values.append(e2)
    
    return pd.DataFrame({
        'Time': times,
        'Exx': strain_xx_values,
        'Eyy': strain_yy_values,
        'Exy': shear_values,
        'e1': e1_values,
        'e2': e2_values
    })

def plot_principal_strains(df, subset_id, settings):
    """Create separate interactive plots for maximum and minimum principal strains."""
    # Create separate figures for e1 and e2
    fig_e1 = go.Figure()
    fig_e2 = go.Figure()
    
    # Add e1 trace
    fig_e1.add_trace(
        go.Scatter(x=df['Time'], y=df['e1'], name="Maximum Principal Strain",
                  line=dict(color='blue'))
    )
    
    # Add e2 trace
    fig_e2.add_trace(
        go.Scatter(x=df['Time'], y=df['e2'], name="Minimum Principal Strain",
                  line=dict(color='red'))
    )
    
    # Update e1 layout
    fig_e1.update_layout(
        title=f'Maximum Principal Strain (e11) Over Time (Subset ID: {subset_id})',
        xaxis_title='Frame Number',
        yaxis_title='Maximum Principal Strain (e11)',
        hovermode='x unified',
        showlegend=True
    )
    
    # Update e2 layout
    fig_e2.update_layout(
        title=f'Minimum Principal Strain (e22) Over Time (Subset ID: {subset_id})',
        xaxis_title='Frame Number',
        yaxis_title='Minimum Principal Strain (e22)',
        hovermode='x unified',
        showlegend=True
    )
    
    # Add grid to both plots
    for fig in [fig_e1, fig_e2]:
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    # Get output paths from settings
    vis_settings = settings.get('visualization_settings', {})
    output_prefix = vis_settings.get('output_prefix', 'principal_strains')
    
    # Save e1 plot
    e1_filename = f"{output_prefix}_e1.html"
    fig_e1.write_html(e1_filename)
    
    # Save e2 plot
    e2_filename = f"{output_prefix}_e2.html"
    fig_e2.write_html(e2_filename)
    
    # Save static versions
    fig_e1.write_image(f"{output_prefix}_e1.png")
    fig_e2.write_image(f"{output_prefix}_e2.png")
    
    print("\nPrincipal strain plots saved as:")
    print(f"- {e1_filename} (interactive)")
    print(f"- {e2_filename} (interactive)")
    print(f"- {output_prefix}_e1.png (static)")
    print(f"- {output_prefix}_e2.png (static)")

def main():
    # Load settings to get results directory
    with open('settings.json', 'r') as f:
        settings = json.load(f)
        results_dir = settings.get('output_folder')
    
    # Find first solution file
    first_solution = find_first_solution_file(results_dir)

    if not first_solution:
        print("No initial solution file found!")

        return
    
    # Get valid subset ID from user
    subset_id = get_valid_id(first_solution)
    if subset_id is None:
        print("Analysis cancelled.")
        return
    
    # Process all solutions and calculate principal strains
    print(f"\nAnalyzing principal strains for subset ID {subset_id}...")
    df = process_all_solutions(results_dir, subset_id)
    
    if df is not None:
        # Create and save plots
        plot_principal_strains(df, subset_id, settings)

if __name__ == '__main__':
    main()
