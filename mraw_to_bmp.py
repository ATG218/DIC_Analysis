import os
import pyMRAW
import numpy as np
from PIL import Image

# Variable for input .cihx file (set this to your file)
cihx_path = r"C:\\Users\\alext\\OneDrive\\Documents\\UROP25\\data\\OneDrive_2025-05-09\\PH 17-4 data\\Noise_floor\\17_4_noisefloor_C001H001S0001.cihx"
output_bmp_dir = r"C:\\Users\\alext\\OneDrive\\Documents\\UROP25\\data\\OneDrive_2025-05-09\\PH 17-4 data\\noisefloorbmp"

# Ensure output directory exists
os.makedirs(output_bmp_dir, exist_ok=True)

# Check if file exists
if not os.path.isfile(cihx_path):
    print(f"Input file does not exist: {cihx_path}")
    exit(1)

# Load video and info using pyMRAW
images, info = pyMRAW.load_video(cihx_path)

frame_count = images.shape[0]
saved_bmp_files = []
timestamps = info['timestamps'] if 'timestamps' in info else [None] * frame_count
cycle_count = 0

for idx in range(frame_count):
    # Start a new folder every 8 frames
    if idx % 8 == 0:
        if idx > 0:
            # Write timestamp.txt for the previous folder
            ts_file_bmp = os.path.join(cycle_folder_bmp, 'timestamp.txt')
            with open(ts_file_bmp, 'w') as f:
                for ts in timestamps_in_cycle:
                    f.write(f"{ts}\n")
            timestamps_in_cycle = []
        cycle_folder_bmp = os.path.join(output_bmp_dir, f't_{cycle_count:04d}')
        os.makedirs(cycle_folder_bmp, exist_ok=True)
        frame_in_cycle = 0
        cycle_count += 1
        timestamps_in_cycle = []
    # Get frame
    frame = images[idx]
    # Convert 16-bit/12-bit to 8-bit for BMP saving
    if frame.dtype == np.uint16 or frame.max() > 255:
        # Clip to 12-bit range if needed, then scale
        frame_8bit = np.clip(frame, 0, 4095)
        frame_8bit = (frame_8bit / 4095 * 255).astype('uint8')
    else:
        frame_8bit = frame.astype('uint8')
    bmp_filename = f"test_{frame_in_cycle}.bmp"
    bmp_path = os.path.join(cycle_folder_bmp, bmp_filename)
    # Save frame as BMP using PIL
    img_bmp = Image.fromarray(frame_8bit)
    img_bmp.save(bmp_path)
    saved_bmp_files.append(bmp_path)
    # Save timestamp for this frame
    ts = timestamps[idx] if timestamps else ''
    timestamps_in_cycle.append(ts)
    frame_in_cycle += 1

# Write timestamp.txt for the last folders (if any frames were extracted)
if frame_count % 8 != 0 or frame_count > 0:
    ts_file_bmp = os.path.join(cycle_folder_bmp, 'timestamp.txt')
    with open(ts_file_bmp, 'w') as f:
        for ts in timestamps_in_cycle:
            f.write(f"{ts}\n")

print(f"Input file: {cihx_path}")
print(f"Output BMP directory: {output_bmp_dir}")
print(f"Total frames extracted and saved: {frame_count}")
print(f"Total folders created: {cycle_count}")
if len(saved_bmp_files) > 0:
    print(f"First 5 BMP files: {saved_bmp_files[:5]}")
else:
    print("No frames were extracted. Please check your .cihx/.mraw files.")
