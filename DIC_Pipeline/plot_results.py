import os
import glob
import json
import pandas as pd
import plotly.graph_objects as go

def read_solution_file(solution_file):
    """Read a DICe solution file and return the data as a DataFrame."""
    try:
        return pd.read_csv(solution_file, skiprows=20, sep=',')
    except Exception as e:
        print(f"Error reading solution file: {e}")
        raise

def find_first_solution_file(results_dir):
    """Find the first DICe solution file in the results directory."""
    # Use glob to find all solution files and sort them
    solution_pattern = os.path.join(results_dir, 'DICe_solution_*.txt')
    solution_files = sorted(glob.glob(solution_pattern))
    return solution_files[0] if solution_files else None

def plot_tracking_points(solution_file, settings):
    """Create an interactive scatter plot of tracking points with their IDs."""
    try:
        # Read the solution file
        df = read_solution_file(solution_file)
        
        # Get visualization settings
        vis_settings = settings.get('visualization_settings', {})
        display_opts = vis_settings.get('display_options', {})
        
        # Create the scatter plot using plotly
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=df['COORDINATE_X'],
            y=df['COORDINATE_Y'],
            mode='markers+text',
            marker=dict(
                size=display_opts.get('point_size', 8),
                color=display_opts.get('point_color', 'blue'),
                opacity=display_opts.get('point_opacity', 0.6)
            ),
            text=df['SUBSET_ID'].astype(int).astype(str),
            textposition="top right",
            textfont=dict(size=display_opts.get('label_size', 8)),
            hovertemplate="ID: %{text}<br>X: %{x}<br>Y: %{y}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title='DICe Tracking Points',
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            yaxis=dict(
                scaleanchor="x",  # Make the aspect ratio 1:1
                scaleratio=1,
            ),
            showlegend=False,
            hovermode='closest',
            plot_bgcolor='white'  # White background
        )
        
        # Add grid if enabled
        if display_opts.get('show_grid', True):
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        # Create output directory if it doesn't exist
        analysis_output_dir = settings.get('output_folder')
        os.makedirs(analysis_output_dir, exist_ok=True)
        
        # Get output prefix from settings or use default
        output_prefix = vis_settings.get('output_prefix', 'tracking_points')
        
        # Save the plots
        html_file = os.path.join(analysis_output_dir, f'{output_prefix}.html')
        png_file = os.path.join(analysis_output_dir, f'{output_prefix}.png')
        
        fig.write_html(html_file)
        fig.write_image(png_file)
        
        print("\nTracking point plots saved as:")
        print(f"- {html_file} (interactive)")
        print(f"- {png_file} (static)")
        
    except Exception as e:
        print(f"Error plotting tracking points: {e}")
        raise

def main():
    # Load settings
    with open('settings.json', 'r') as f:
        settings = json.load(f)
    
    # Find the solution file
    solution_directory = settings.get('output_folder') + "/results/"
    solution_file = find_first_solution_file(solution_directory)
    if solution_file:
        try:
            plot_file = plot_tracking_points(solution_file, settings)
            print(f"Plot saved as: {plot_file}")
        except Exception as e:
            print(f"Failed to create plot: {str(e)}")
    else:
        print("No solution file found in results directory")

if __name__ == "__main__":
    main()
