import json
import os

def generate_json_config(dice_exe_path, reference_path, input_folder, output_base_folder, subset_size, step_size, strain_window_size, output_path):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Base structure of the JSON
    config = {
        "dice_exe_path": dice_exe_path,
        "reference_path": reference_path,
        "input_folder": input_folder,
        "output_folder": "{}/DICe_subset{:02}_step{:02}_VSG{:03}".format(output_base_folder, subset_size, step_size, strain_window_size),
        "subset_size": subset_size,
        "step_size": step_size,
        "analysis_settings": {
            "sssig_threshold": 30,
            "enable_translation": True,
            "enable_rotation": True,
            "enable_normal_strain": True,
            "enable_shear_strain": True,
            "optimization_method": "GRADIENT_BASED",
            "initialization_method": "USE_FEATURE_MATCHING",
            "output_delimiter": ",",
            "strain_window_size": strain_window_size
        },
        "output_spec": {
            "COORDINATE_X": True,
            "COORDINATE_Y": True,
            "VSG_STRAIN_XX": True,
            "VSG_STRAIN_YY": True,
            "VSG_STRAIN_XY": True
        },
        "region_of_interest": {
            "type": "polygon",
            "vertices": [
[200, 102],
[342, 113],
[342, 118],
[342, 124],
[342, 133],
[342, 143],
[342, 154],
[342, 165],
[342, 175],
[342, 184],
[341, 196],
[341, 207],
[341, 217],
[341, 228],
[341, 239],
[341, 250],
[342, 260],
[345, 271],
[349, 282],
[355, 293],
[365, 302],
[374, 312],
[385, 318],
[394, 323],
[406, 326],
[417, 326],
[427, 325],
[438, 324],
[449, 322],
[461, 318],
[470, 311],
[480, 304],
[490, 294],
[496, 284],
[501, 273],
[506, 262],
[506, 251],
[506, 241],
[507, 229],
[506, 219],
[506, 209],
[506, 198],
[506, 186],
[505, 176],
[505, 165],
[504, 155],
[504, 144],
[504, 134],
[504, 123],
[503, 118],
[500, 114],
[822, 100],
[824, 622],
[202, 624]

        ]
        },
        "visualization_settings": {
            "plot_type": "scatter",
            "data_columns": {
                "x": "COORDINATE_X",
                "y": "COORDINATE_Y",
                "id": "SUBSET_ID"
            },
            "display_options": {
                "show_ids": True,
                "point_size": 8,
                "point_color": "blue",
                "point_opacity": 0.6,
                "label_size": 8,
                "show_grid": True
            },
            "save_formats": ["html", "png"]
        },
        "settings_paths": {}
    }
    
    # Write the JSON file
    with open(output_path, 'w') as json_file:
        json.dump(config, json_file, indent=4)
    
   # print(f"Configuration file saved to {output_path}")

def generate_summary_json(settings_files, subset_ids, summary_output_path):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)
    
    summary_config = {
        "settings_files": settings_files,
        "subset_ids": subset_ids,
        "parallel_processing": True
    }
    
    with open(summary_output_path, 'w') as json_file:
        json.dump(summary_config, json_file, indent=4)
    
    print(f"Summary configuration file saved to {summary_output_path}")



# Generate file(s)
dice_exe_path="C:/Program Files (x86)/Digital Image Correlation Engine/dice.exe"

MainPath = "C:/Users/METALS/Documents/Fatigue rig/Experiments/20250306_Heating test/Noise floor room temperature_sorted/Notch_"
NotchNum = "09"

reference_path=MainPath+NotchNum+"/Notch_"+NotchNum+"_cycle_00001_00001.bmp"
input_folder=MainPath+NotchNum+"/"
output_folder=MainPath+NotchNum+"/DICe/"

subset_list = [21,25,29]
step_list = [3]

# VSGs are multiples of step_list, where the strain_guage is vsg_list * step_list
vsg_list = [6,9,12,15,18,21,24,27]

# Preallocate list
settings_files = []

for subset_size in subset_list:
    for step_size in step_list:
        for vsg in vsg_list:
            # print(vsg)
            filename = "C:/Users/METALS/Documents/DIC_Analysis/temp/n"+NotchNum+"_sub{:02}_stp{:02}_vsg{:03}.json".format(subset_size, step_size, vsg)
            generate_json_config(dice_exe_path, reference_path, input_folder, output_folder, subset_size, step_size, vsg, filename)
            settings_files.append(filename)

# Generate the batch settings JSON file
subset_ids = []  # Example subset IDs

summary_filename = "C:/Users/METALS/Documents/DIC_Analysis/temp/batch_settings.json"  # Change as needed
generate_summary_json(settings_files, subset_ids, summary_filename)