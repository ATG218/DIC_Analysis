# Notch Parser

This project processes a dataset of notches based on a set of reference images extracted from the data. The script will automatically create a reference directory if faster storage is preferable and will output 8 folders with the sorted notches within them. Notches are irrelevant to size and may differ between datasets

## How It Works

1. **Reference Image Processing**:
   - The script first analyzes a set of reference images from the dataset
   - For each reference image, it:
     - Detects bright regions (pixel values > 240) in the upper portion of the image
     - Identifies continuous regions while allowing small gaps (up to 5 pixels)
     - Measures the width of each detected notch
     - Creates a mapping between original notch positions and their sorted order

2. **Notch Width Detection**:
   - Scans a defined search area (starting at 25% from the top of the image)
   - Creates a mask of bright pixels (value > 240)
   - Finds continuous regions of bright pixels, allowing small gaps
   - Measures the width of each region
   - Validates measurements against minimum and maximum width constraints

3. **Dataset Organization**:
   - Sorts notches by width from smallest to largest
   - Assigns each width a notch number (1-8)
   - Creates a mapping between original positions and sorted notch numbers

## Output Format

The script generates the following outputs in the specified output directory:

1. **Sorted Folders**:
   - Creates 8 folders named `Notch_1` through `Notch_8`
   - Notch_1 contains images with the smallest notch width
   - Notch_8 contains images with the largest notch width

2. **Debug Images** (during processing):
   - `debug_notch_width.png`: Shows detected notch regions with green lines
   - `debug_edges.png`: Displays the bright pixel mask used for detection

3. **Timestamps File**:
   - Creates `timestamps.txt` in the output directory
   - Contains chronological processing information for each image
   - Format: `[timestamp] [cycle_number] [image_name]`

4. **Log File**:
   - Records detailed processing information including:
     - Notch width measurements
     - Mapping between original and sorted positions
     - Processing status for each subfolder
     - Error messages and warnings
     - Total processing time

## Requirements

- Python 3.6+
- `opencv-python`
- `PyInstaller`

## Installation

1. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Create a `settings.txt` file in the root directory with the following format:
    ```
    input_directory=path/to/input_directory
    output_directory=path/to/output_directory
    log_directory=path/to/log_file.log
    reference_directory=path/to/reference_directory
    ```

    - `input_directory`: Path to the directory containing the input subfolders with images.
    - `output_directory`: Path to the directory where the sorted images will be saved.
    - `log_directory`: Path to the log file where processing information will be logged INCLUDING {FILENAME}.log
    - `reference_directory`: Path to the directory where reference images will be temporarily stored.

2. Run the script:
    ```sh
    python main.py
    ```

## Packaging as an Executable

To package this project as an executable, you can use `PyInstaller`.

1. Install `PyInstaller`:
    ```sh
    pip install pyinstaller
    ```

2. Create a `spec` file for the project (optional but recommended):
    ```sh
    pyinstaller --name notch_parser --onefile src/main.py
    ```

3. Build the executable:
    ```sh
    pyinstaller notch_parser.spec
    ```

4. The executable will be created in the `dist` directory.

## Logging

The script maintains detailed logs of the processing pipeline:
- Settings configuration and validation
- Notch width measurements and mapping
- Folder creation and file operations
- Processing status for each image/subfolder
- Error handling and warnings
- Performance metrics (processing time)

## Deleting Reference Directory

After processing, the script deletes the reference directory specified in the `settings.txt` file. This is only a temporary folder to meet the case of slower external storage. The path will only be used for comparison and then deleted after sorting completion

## Timestamps

The timestamps.txt file provides a chronological record of image processing:
- Each line contains: timestamp, cycle number, and image name
- Helps track processing order and timing
- Useful for verification and debugging purposes