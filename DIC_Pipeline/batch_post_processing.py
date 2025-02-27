import logging
import os
import json
from pathlib import Path
import plotly.graph_objects as go
import pandas as pd
from io import StringIO
import glob
from strain_analysis import process_all_solutions, calculate_principal_strains
import time
import datetime

def setup_logging():
    """Set up logging configuration."""
    try:
        # Get output directory from first settings file in batch_settings
        with open('batch_settings.json', 'r') as f:
            batch_settings = json.load(f)
            settings_files = batch_settings.get('settings_files', [])
            
            if settings_files:
                with open(settings_files[0], 'r') as sf:
                    settings = json.load(sf)
                output_path = Path(settings['output_folder'])
                # Create parent directory first
                output_path.parent.mkdir(parents=True, exist_ok=True)
                log_dir = output_path.parent / 'logs'
                log_dir.mkdir(exist_ok=True)
                log_file = log_dir / 'batch_post_processing.log'
            else:
                # Fallback to default logs directory
                log_file = Path('logs') / 'batch_post_processing.log'
                log_file.parent.mkdir(exist_ok=True)
    except Exception as e:
        # Fallback to default logs directory
        log_file = Path('logs') / 'batch_post_processing.log'
        log_file.parent.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def get_output_directory(settings_files, create_point_maps=False):
    """Get the output directory from the first settings file's parent directory."""
    try:
        with open(settings_files[0], 'r') as f:
            first_settings = json.load(f)
        output_path = Path(first_settings['output_folder'])
        parent_dir = output_path.parent
        
        # Only create point maps directory if specifically requested
        if create_point_maps:
            point_maps_dir = parent_dir / 'pointmaps'
            point_maps_dir.mkdir(exist_ok=True)
            
        strain_analysis_dir = parent_dir / 'strain_analysis'
        strain_analysis_dir.mkdir(exist_ok=True)
        
        return parent_dir
    except Exception as e:
        logger.error(f"Error getting output directory: {str(e)}")
        # Fallback to current directory
        return Path('.')

def process_settings_file(settings_file, subset_id, logger):
    """Process a single settings file and return strain data."""
    try:
        # Load settings
        with open(settings_file, 'r') as f:
            settings = json.load(f)
            
        # Get subset size from settings
        subset_size = settings.get('subset_size', 'unknown')
            
        # Process strain data
        strain_data = process_all_solutions(settings['output_folder'], subset_id)
        
        if strain_data is None:
            raise Exception(f"No strain data found for subset {subset_id}")
        
        # Save strain data to CSV
        output_dir = get_output_directory([settings_file])
        strain_dir = output_dir / 'strain_analysis'
        strain_dir.mkdir(exist_ok=True)
        
        # Extract notch number from input folder path
        input_folder = Path(settings['input_folder'])
        notch_name = input_folder.name.lower()  # e.g., "Notch_02"
        
        # Create CSV filename with subset size
        csv_filename = f"{notch_name}_subset{subset_id}_{settings['analysis_settings']['strain_window_size']}_{subset_size}_principal_data.csv"
        csv_path = strain_dir / csv_filename
        
        # Save to CSV
        strain_data.to_csv(csv_path, index=False)
        logger.info(f"Strain data saved to CSV: {csv_path}")
        
        return {
            'name': f'{notch_name}',
            'data': strain_data,
            'subset_id': subset_id,
            'strain_size': settings['analysis_settings']['strain_window_size'],
            'subset_size': settings['subset_size'],
            'step_size': settings['step_size']
        }
        
    except Exception as e:
        logger.error(f"Error processing {settings_file}: {str(e)}")
        raise



def create_combined_strain_plot(all_strains, output_dir, logger):
    """Create separate interactive plots for e1 and e2 strain data."""
    try:
        # Color palette for different notches
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        # Create figures for e1 and e2
        fig_e1 = go.Figure()
        fig_e2 = go.Figure()
        
        # Add traces for each settings file
        for i, strain_data in enumerate(all_strains):
            if strain_data is None:
                continue
                
            name = strain_data['name']
            df = strain_data['data']
            subset_id = strain_data['subset_id']
            strain_size = strain_data['strain_size']
            subset_size = strain_data['subset_size']
            color = colors[i % len(colors)]
            
            # Add e1 (max principal strain) to e1 plot
            fig_e1.add_trace(go.Scatter(
                x=df['Time'],
                y=df['e1'],
                name=f"{name} (ID: {subset_id}, Subset: {subset_size}, Strain: {strain_size})",
                line=dict(color=color),
                mode='lines',
                visible=True
            ))
            
            # Add e2 (min principal strain) to e2 plot
            fig_e2.add_trace(go.Scatter(
                x=df['Time'],
                y=df['e2'],
                name=f"{name} (ID: {subset_id}, Subset: {subset_size}, Strain: {strain_size})",
                line=dict(color=color),
                mode='lines',
                visible=True
            ))

        # Update e1 layout
        fig_e1.update_layout(
            title="Maximum Principal Strain (e11) Analysis",
            xaxis_title="Frame Number",
            yaxis_title="Maximum Principal Strain (e11)",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05,
                itemclick='toggle',
                itemdoubleclick='toggleothers'
            ),
            width=1000,
            height=600,
            margin=dict(r=200)
        )
        
        # Update e2 layout
        fig_e2.update_layout(
            title="Minimum Principal Strain (e22) Analysis",
            xaxis_title="Frame Number",
            yaxis_title="Minimum Principal Strain (e22)",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05,
                itemclick='toggle',
                itemdoubleclick='toggleothers'
            ),
            width=1000,
            height=600,
            margin=dict(r=200)
        )
        
        # Add grid to both plots
        for fig in [fig_e1, fig_e2]:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        # Save e1 plot
        e1_plot_file = output_dir / 'max_principal_strain_analysis.html'
        fig_e1.write_html(str(e1_plot_file))
        logger.info(f"Maximum principal strain plot saved to: {e1_plot_file}")
        
        # Save e2 plot
        e2_plot_file = output_dir / 'min_principal_strain_analysis.html'
        fig_e2.write_html(str(e2_plot_file))
        logger.info(f"Minimum principal strain plot saved to: {e2_plot_file}")
        
        # Save as PNG for quick reference
        fig_e1.write_image(str(output_dir / 'max_principal_strain_analysis.png'))
        fig_e2.write_image(str(output_dir / 'min_principal_strain_analysis.png'))
        logger.info("Static plots saved as PNG files")
        
    except Exception as e:
        logger.error(f"Error creating strain plots: {str(e)}")
        raise

def find_first_solution_file(results_dir):
    """Find the first DICe solution file in the results directory."""
    # Use glob to find all solution files and sort them
    solution_pattern = os.path.join(results_dir, "DICe_solution_*.txt")
    solution_files = sorted(glob.glob(solution_pattern))
    return solution_files[0] if solution_files else None

def generate_point_map(settings_file, logger):
    """Generate point map for a settings file."""
    try:
        with open(settings_file, 'r') as f:
            settings = json.load(f)
        
        output_dir = get_output_directory([settings_file], create_point_maps=True)
        point_maps_dir = output_dir / 'pointmaps'
        
        # Find first solution file
        first_file = None
        results_dir = os.path.join(settings['output_folder'], "results")
        if os.path.exists(results_dir):
            first_file = find_first_solution_file(results_dir)
        
        if not first_file:
            logger.error(f"No solution files found in {results_dir}")
            return
            
        # Read the first solution file
        df = pd.read_csv(first_file, skiprows=20)
        #df = pd.read_csv(first_file)
        # Create scatter plot of points
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['COORDINATE_X'],
            y=df['COORDINATE_Y'],
            mode='markers+text',
            marker=dict(
                size=8,
                color='blue',
                opacity=0.6
            ),
            text=df['SUBSET_ID'].astype(int).astype(str),
            textposition="top right",
            textfont=dict(size=8),
            hovertemplate="ID: %{text}<br>X: %{x}<br>Y: %{y}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Point Map for {os.path.basename(settings_file)}',
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            yaxis=dict(
                scaleanchor="x",  # Make the aspect ratio 1:1
                scaleratio=1,
            ),
            showlegend=False,
            hovermode='closest',
            plot_bgcolor='white'
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        # Save plot
        base_name = os.path.splitext(os.path.basename(settings_file))[0]
        html_file = point_maps_dir / f'{base_name}_point_map.html'
        
        fig.write_html(str(html_file))
        logger.info(f"Point map saved to: {html_file}")
        
    except Exception as e:
        logger.error(f"Error generating point map for {settings_file}: {str(e)}")

def batch_post_process(settings_file, subset_id, logger):
    """Process a single settings file for batch processing."""
    start_time = time.time()
    logger.info(f"Starting post-processing for {settings_file} at {datetime.datetime.now()}")
    try:
        # Process the settings file
        strain_data = process_settings_file(settings_file, subset_id, logger)
        if strain_data is None:
            raise Exception(f"No strain data found for subset {subset_id}")
            
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Post-processing completed for {settings_file} at {datetime.datetime.now()}")
        logger.info(f"Processing time for {settings_file}: {duration:.2f} seconds")
        return strain_data
        
    except Exception as e:
        logger.error(f"Error processing settings: {str(e)}")
        raise

def process_all_settings(settings_files, subset_ids, logger):
    """Process all settings files and create combined plot."""
    start_time = time.time()
    logger.info(f"Starting batch post-processing at {datetime.datetime.now()}")
    
    all_strains = []
    output_dir = get_output_directory(settings_files)
    
    # Process each settings file
    for settings_file, subset_id in zip(settings_files, subset_ids):
        try:
            logger.info(f"Processing {settings_file} with subset ID {subset_id}")
            strain_data = batch_post_process(settings_file, subset_id, logger)
            all_strains.append(strain_data)
        except Exception as e:
            logger.error(f"Error processing {settings_file}: {str(e)}")
            
    # Create combined plot
    if all_strains:
        try:
            strain_analysis_dir = output_dir / 'strain_analysis'
            create_combined_strain_plot(all_strains, strain_analysis_dir, logger)
        except Exception as e:
            logger.error(f"Error creating combined plot: {str(e)}")
            
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Batch post-processing completed at {datetime.datetime.now()}")
    logger.info(f"Total batch post-processing time: {duration:.2f} seconds")

def main():
    logger = setup_logging()
    logger.info("Starting batch post-processing")
    print("Starting batch post-processing")
    
    try:
        # Load batch settings
        with open('batch_settings.json', 'r') as f:
            batch_settings = json.load(f)
        
        settings_files = batch_settings.get('settings_files', [])
        subset_ids = batch_settings.get('subset_ids', [])
        
        if not settings_files:
            error_msg = "No settings files specified in batch_settings.json"
            logger.error(error_msg)
            print(error_msg)
            return
        
        if not subset_ids:
            info_msg = "No subset IDs specified, generating point maps..."
            logger.info(info_msg)
            print(info_msg)
            
            for settings_file in settings_files:
                generate_point_map(settings_file, logger)
           
            final_msg = "\nPlease add the desired subset IDs to batch_settings.json and run batch post-processing again."
            logger.info(final_msg)
            print(final_msg)
               
            return
            
        # Process settings files with subset IDs
        process_all_settings(settings_files, subset_ids, logger)
        
        logger.info("Batch post-processing completed")
        print("Batch post-processing completed")
        
    except Exception as e:
        logger.error(f"Batch post-processing failed: {str(e)}")
        print(f"Batch post-processing failed: {str(e)}")

if __name__ == "__main__":
    main()
