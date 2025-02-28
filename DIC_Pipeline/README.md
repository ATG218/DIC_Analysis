# DICe Processing Scripts

This repository contains a set of Python scripts for running Digital Image Correlation (DICe) analysis and post-processing the results. This pipeline is designed to be run after using the notch_parser sorting script. Test

## ⚠️ Important Requirements

Before using these scripts, ensure:
1. All images are in `.bmp` format
2. Input folder paths in settings.json MUST end with a forward slash (`/`)
3. DICe is installed on your system
4. Python 3.7+ is installed
5. Required Python packages are installed (run `pip install -r requirements.txt`)

## Settings Structure

The settings files are the heart of the system. There are two ways to use them:
1. Single analysis: Use `settings.json` with `dice_processing.py`
2. Batch analysis: Use multiple settings files (any name) with `batch_processing.py`

### Settings File Structure (CRITICAL REQUIREMENTS)

```json
{
    "dice_exe_path": "C:/Program Files (x86)/Digital Image Correlation Engine/dice.exe",
    "reference_path": "C:/path/to/images/Notch_01/Notch_01_cycle_0000_00001.bmp",
    "input_folder": "C:/path/to/images/Notch_01/",  // MUST end with '/'
    "output_folder": "C:/path/to/output/20250129_304V4/notch1",
    "subset_size": 25,  // Size of the correlation window
    "step_size": 5,    // Distance between correlation points
    "analysis_settings": {
        "sssig_threshold": 500,      // Signal-to-noise ratio threshold
        "enable_translation": true,   // Allow translation motion
        "enable_rotation": true,      // Allow rotation motion
        "enable_normal_strain": true, // Calculate normal strain
        "enable_shear_strain": true,  // Calculate shear strain
        "strain_window_size": 30      // Size of strain calculation window
    },
    "visualization_settings": {
        "output_prefix": "notch1",    // Prefix for output files
        "display_options": {
            "point_size": 8,          // Size of tracking points in plots
            "point_color": "blue",     // Color of tracking points
            "point_opacity": 0.6,      // Transparency of points
            "label_size": 8           // Size of point labels
        }
    }
}
```

### Critical Settings Requirements

1. **File Paths**:
   - `dice_exe_path`: Must point to your DICe installation
   - `reference_path`: Must be a `.bmp` file
   - `input_folder`: Must end with a forward slash (`/`)
   - All paths must use forward slashes (`/`) not backslashes (`\`)

2. **Image Requirements**:
   - All images must be in `.bmp` format
   - Reference image should be the first frame (usually `*_00001.bmp`)
   - Images should be named sequentially

3. **Analysis Parameters**:
   - `subset_size`: Must be odd number (recommended: 25-31)
   - `step_size`: Typically 1/5 of subset_size
   - `sssig_threshold`: Adjust based on image quality (higher = stricter)

### Batch Processing Settings

For batch processing, create a `batch_settings.json`:

```json
{
    "settings_files": [
        "settings1.json",
        "settings2.json",
        "settings3.json"
    ],
    "subset_ids": [246, 246, 246],  // One ID per settings file
    "parallel_processing": true     // Enable parallel processing
}
```

- Each settings file in the list will be processed
- Subset IDs must match the order of settings files
- Files can have any name (not just settings1.json, etc.)
- Output folders in each settings file must be unique

## Output Structure

The scripts generate several types of outputs:

1. **DICe Results** (in output_folder/results/):
   - `DICe_solution_*.txt`: Raw correlation results
   - Contains displacement and strain data

2. **Strain Analysis** (in output_folder/strain_analysis/):
   - `*_e11.html/png`: Maximum principal strain plots
   - `*_e22.html/png`: Minimum principal strain plots
   - `*_subset_*_principal_data.csv`: Strain data in CSV format

3. **Logs** (in logs/):
   - Detailed processing logs with timestamps
   - Error messages and warnings

## Output File Structure

### Regular Processing (using `dice_processing.py` and `dice_post_processing.py`)
When using the regular processing pipeline with a single `settings.json`, your output directory will be organized as follows:

```
output_folder/                           # Specified in settings.json
├── logs/                               # All processing logs
│   ├── dice_workflow_*.log            # DICe processing logs
│   ├── dice_generate_*.log            # Configuration generation logs
│   └── dice_post_processing_*.log     # Post-processing logs
├── results/                           # DICe analysis results
│   └── DICe_solution_*.txt           # Raw DICe output files
├── input.xml                         # DICe input configuration
├── params.xml                        # DICe parameters
├── subsets.txt                       # Region of interest definition
├── [folder_name]_tracking_points.html  # Interactive tracking visualization
├── [folder_name]_tracking_points.png   # Static tracking visualization
├── subset_[id]_principal_strains_e11.html  # Max principal strain plot
├── subset_[id]_principal_strains_e11.png   # Static max strain plot
├── subset_[id]_principal_strains_e22.html  # Min principal strain plot
├── subset_[id]_principal_strains_e22.png   # Static min strain plot
└── subset_[id]_principal_data.csv          # Strain data CSV
```

### Batch Processing (using `batch_processing.py`)
When using batch processing with multiple settings files, each output directory will follow this structure:

```
parent_output_folder/                    # Parent directory for all analyses
├── logs/                               # Batch processing logs
│   ├── batch_processing_*.log         # Main batch processing logs
│   └── batch_post_processing_*.log    # Batch post-processing logs
├── notch1/                             # From settings1.json
│   ├── results/
│   │   └── DICe_solution_*.txt
│   ├── input.xml
│   ├── params.xml
│   ├── subsets.txt
├── notch2/                             # From settings2.json
│   ├── [same structure as notch1]
└── .../                             
    └── [same structure as notch1]
└── pointmaps/                           # From post-processing         
    └── [settingsFile]_point_map.html
    └── ...
└── strain_analysis/                    # From post-processing   
    └── max_principal_strain_analysis.html
    └── max_principal_strain_analysis.png
    └── min_principal_strain_analysis.html
    └── min_principal_strain_analysis.png
    └── [settingsFile]_subset_[id]_principal_data.csv
    └── ...
```


### Important Notes About File Structure

1. **Naming Conventions**:
   - `[folder_name]`: Used only for tracking point visualizations
   - `[id]`: The subset ID used for strain analysis
   - `*.log`: Includes timestamp in format `YYYYMMDD_HHMMSS`

2. **File Types**:
   - `.html`: Interactive plots viewable in web browsers
   - `.png`: Static images for reports/publications
   - `.csv`: Raw data for further analysis
   - `.txt`: DICe solution files and configuration
   - `.xml`: DICe input and parameter files

3. **Organization**:
   - `logs/`: Contains all processing logs in one place
   - `results/`: Contains raw DICe output files
   - Root directory: Contains all visualization files and processed data

4. **Batch Processing**:
   - Each settings file gets its own directory
   - All files for one analysis stay in their respective directory
   - Batch logs are stored in parent directory
   - Easy to process multiple notches/samples in parallel

## Common Issues

1. **Path Issues**:
   - Input folder missing trailing slash
   - Using backslashes instead of forward slashes
   - Spaces in file paths

2. **Image Issues**:
   - Images not in .bmp format
   - Reference image not found
   - Inconsistent image naming

3. **Analysis Issues**:
   - `sssig_threshold` too high/low
   - `subset_size` too small/large
   - Output folder permissions

## Tips for Best Results

1. **Image Quality**:
   - Use high-contrast images
   - Ensure good speckle pattern
   - Avoid motion blur

2. **Parameter Selection**:
   - Start with default subset_size (25)
   - Adjust sssig_threshold based on correlation success
   - Use smaller step_size for finer resolution

3. **Performance**:
   - Enable parallel_processing for batch analysis
   - Keep subset_size reasonable (25-31)
   - Monitor disk space in output folder

## Workflow Overview

The workflow is split into two main phases:
1. Processing (`dice_processing.py`): Generates configuration files and runs DICe analysis
2. Post-processing (`dice_post_processing.py`): Creates visualizations and strain analysis

### Script Details

#### 1. DICe Processing (`dice_processing.py`)
- **Purpose**: Generate configuration files and run DICe analysis
- **When to use**: Run this first to process your images
- **Features**:
  - Checks for existing configuration files
  - Option to use existing files or generate new ones
  - Prompts before running DICe analysis
- **Generated Files**:
  - `input.xml`: DICe input configuration
  - `params.xml`: Analysis parameters
  - `subsets.txt`: Region of interest definition
- **Logging**:
  - Location: `logs/dice_workflow_YYYYMMDD_HHMMSS.log`
  - Logs: File generation, DICe execution, errors

#### 2. Post-Processing (`dice_post_processing.py`)
- **Purpose**: Generate visualizations and strain analysis
- **When to use**: After DICe processing is complete
- **Features**:
  - Tracking point visualization
  - Principal strain analysis
  - CSV data export
- **Generated Files**:
  - `{folder_name}_tracking_points.html/png`: Interactive/static tracking visualization
  - `{folder_name}_subset_{id}_principal_strains.html/png`: Strain plots
  - `{folder_name}_subset_{id}_strain_data.csv`: Strain data
- **Logging**:
  - Location: `logs/dice_post_processing_YYYYMMDD_HHMMSS.log`
  - Logs: File generation, analysis steps, data statistics

### Support Scripts

#### Generate DICe Files (`generate_dice_files.py`)
- **Purpose**: Create DICe configuration files
- **Called by**: `dice_processing.py`
- **Generated Files**:
  - `input.xml`: Image and subset configuration
  - `params.xml`: Analysis parameters
  - `subsets.txt`: Region of interest definition

#### Plot Results (`plot_results.py`)
- **Purpose**: Create tracking point visualizations
- **Called by**: `dice_post_processing.py`
- **Features**:
  - Interactive HTML plots
  - Static PNG images
  - Customizable point display

#### Strain Analysis (`strain_analysis.py`)
- **Purpose**: Calculate and visualize principal strains
- **Called by**: `dice_post_processing.py`
- **Features**:
  - Principal strain calculation
  - Interactive strain plots
  - Data export to CSV

## Batch Processing

The system supports batch processing of multiple analyses using different settings files. This is configured through `batch_settings.json`.

### Batch Settings Configuration

```json
{
    "settings_files": [
        "settings_notch01.json",
        "settings_notch02.json"
    ],
    "subset_ids": [
        "560",
        "561"
    ],
    "parallel_processing": false
}
```

- `settings_files`: List of settings.json files to process
- `subset_ids`: List of subset IDs for post-processing, must match the order of settings_files
- `parallel_processing`: Future support for parallel processing (currently not implemented)

### Running Batch Processing

1. Create individual settings.json files for each analysis you want to run
2. List these files in batch_settings.json along with their corresponding subset IDs
3. Run the batch processing script:
   ```bash
   python batch_processing.py
   ```

### Batch Processing Features

- **Conflict Detection**: The script checks for conflicts in input/output folders and warns you before proceeding
- **Post-Processing Integration**: Option to automatically run post-processing after each DICe analysis
- **Logging**: Comprehensive logging to both console and file:
  - `batch_processing.log`: Main processing log
  - `post_processing.log`: Post-processing specific log

### Output Structure

For each settings file, the script will:
1. Generate DICe input files in the specified output folder
2. Run DICe analysis
3. If post-processing is enabled:
   - Create strain plots (`subset_[ID]_principal_strains.html`)
   - Save processed data (`subset_[ID]_processed_strains.csv`)

### Error Handling

- The script validates all settings files before processing
- Checks that the number of settings files matches the number of subset IDs
- Individual processing failures don't stop the entire batch
- All errors are logged with detailed information

## Configuration (settings.json)

### Basic Settings
```json
{
    "dice_exe_path": "C:/Program Files (x86)/Digital Image Correlation Engine/dice.exe",
    "input_folder": "path/to/images/folder/",
    "reference_path": "path/to/reference/image.bmp",  // Optional: specific reference image
    "output_folder": "path/to/output",
    "subset_size": 27,
    "step_size": 3
}
```

- `dice_exe_path`: Path to DICe executable
- `input_folder`: Directory containing the image sequence
- `reference_path`: Optional path to specific reference image. If provided, this image will be used as the reference instead of the first image in the sequence
- `output_folder`: Directory for analysis outputs
- `subset_size`: Size of correlation subset in pixels
- `step_size`: Step size between subset centers

### Reference Image Handling

When using a `reference_path` in settings.json:
1. The script will automatically copy the reference image to your input folder
2. The copied image will be renamed by adding "_reference" before the file extension
   - Example: "image.bmp" becomes "image_reference.bmp"
3. The copied reference image will be used as the reference for DICe processing
4. All original images remain untouched
5. The script logs all copy operations and any potential errors

Example:
```json
{
    "reference_path": "path/to/reference/image.bmp",
    "input_folder": "path/to/images"
}
```

This will copy "image.bmp" to "path/to/images/image_reference.bmp" and use it as the reference.

### Region of Interest Settings

The `region_of_interest` in settings.json can be configured in two ways:

1. Rectangle Mode:
```json
"region_of_interest": {
    "type": "rectangle",
    "center": [420, 455],
    "width": 125,
    "height": 100
}
```

2. Polygon Mode:
```json
"region_of_interest": {
    "type": "polygon",
    "vertices": [
        [341, 414],
        [541, 418],
        [539, 541],
        [347, 532]
    ]
}
```

### Analysis Settings

Key analysis parameters that can be configured:

```json
"analysis_settings": {
    "sssig_threshold": 150,
    "optimization_method": "GRADIENT_BASED",  // Options: "GRADIENT_BASED", "SIMPLEX"
    "initialization_method": "USE_FEATURE_MATCHING",  // Options: "USE_FEATURE_MATCHING", "USE_NEIGHBOR_VALUES"
    "strain_window_size": 15,
    "enable_translation": true,
    "enable_rotation": true,
    "enable_normal_strain": true,
    "enable_shear_strain": true
}
```

## Typical Workflow

1. Configure `settings.json` with your parameters
2. Run processing:
   ```bash
   python dice_processing.py
   ```
   - Responds to prompts about file generation and analysis
   - Check logs for progress and errors

3. Run post-processing:
   ```bash
   python dice_post_processing.py
   ```
   - Select points for strain analysis
   - Check output directory for results
   - Review logs for analysis details

## Error Handling

- All scripts use detailed logging
- Logs are timestamped and stored in the `logs` directory
- Check logs for:
  - File generation status
  - Analysis progress
  - Error messages and stack traces
  - Data statistics and validation
