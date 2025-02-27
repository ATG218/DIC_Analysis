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
            "sssig_threshold": 250,
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
            [
                280,
                384
            ],
            [
                403,
                387
            ],
            [
                401,
                427
            ],
            [
                411,
                460
            ],
            [
                423,
                489
            ],
            [
                448,
                512
            ],
            [
                485,
                530
            ],
            [
                512,
                536
            ],
            [
                541,
                534
            ],
            [
                582,
                512
            ],
            [
                604,
                492
            ],
            [
                620,
                463
            ],
            [
                624,
                451
            ],
            [
                628,
                390
            ],
            [
                750,
                387
            ],
            [
                751,
                620
            ],
            [
                280,
                620
            ]
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
reference_path="C:/Users/alext/OneDrive/Documents/UROP25/DiceCLI/DICe_Processing/parameterStudy/photos/n5/Notch_05_cycle_0000_00001.bmp"
input_folder="C:/Users/alext/OneDrive/Documents/UROP25/DiceCLI/DICe_Processing/parameterStudy/photos/n5"
output_folder="C:/Users/alext/OneDrive/Documents/UROP25/DiceCLI/DICe_Processing/parameterStudy/guageStudy/n5"

subset_list = [27]
step_list = [2]

# VSGs are multiples of step_list, where the strain_guage is vsg_list * step_list
vsg_list = [4]

settings_files = []

for subset_size in subset_list:
    for step_size in step_list:
        for vsg in [x * step_size for x in vsg_list]:
            print(vsg)
            filename = "guageStudy/temp/n5_sub{:02}_stp{:02}_vsg{:03}.json".format(subset_size, step_size, vsg)
            generate_json_config(dice_exe_path, reference_path, input_folder, output_folder, subset_size, step_size, vsg, filename)
            settings_files.append(filename)

# Generate the batch settings JSON file
subset_ids = []  # Example subset IDs

summary_filename = "guageStudy/temp/batch_settings.json"  # Change as needed
generate_summary_json(settings_files, subset_ids, summary_filename)