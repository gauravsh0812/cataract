import os
import math
import json
import pandas as pd
import tqdm

def seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def get_key_by_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    raise ValueError(f"Value {target_value} not found in dictionary")

def add_value_to_dict(dictionary, new_value):
    # Find the next available integer key
    next_key = max(dictionary.keys()) + 1

    # Add the new value to the dictionary
    dictionary[next_key] = new_value
    return dictionary

def extract_frames(root, case, start, end, phase_dict, phase):
    # Convert start and end times to HH:MM:SS format
    start_hms = seconds_to_hms(start)
    end_hms = seconds_to_hms(end)

    print("start, end: ", start_hms, end_hms, phase)
    
    # Ensure the frames directory exists
    frames_dir = f"{root}/frames/{case}"
    os.makedirs(frames_dir, exist_ok=True)

    category = get_key_by_value(phase_dict, phase)
    
    # Create the ffmpeg command
    cmd = f"ffmpeg -ss {start_hms} -to {end_hms} \
            -i {root}/videos/{case}.mp4 \
            -vf 'fps=1' {frames_dir}/frame_%04d_{start}_{end}_{category}.jpg"
    
    # Execute the command
    os.system(cmd)
    
if __name__ == "__main__":
    root = "/data/shared/cataract-1K/phase_recognition"
    cases = os.listdir(f"{root}/annotations")

    phase_dict = {
        0:"Incision",
        1:"Viscoelastic",
        2:"Capsulorhexis",
        3:"Hydrodissection",
        4:"Phacoemulsification",
        5:"Irrigation/Aspiration",
        6:"Capsule Pulishing",
        7:"Viscoelastic",
        8:"Lens Implantation",
        9:"Lens positioning",
        10:"Viscoelastic_Suction",
        11:"Tonifying/Antibiotics",
        12: "Anterior_Chamber Flushing"
    }

    # for c in cases:
    #     if c != "SYNAPSE_METADATA_MANIFEST.tsv":
    #         f = pd.read_csv(f"{root}/annotations/{c}/{c}_annotations_phases.csv")
    #         for _, row in f.iterrows():
    #             start, end = math.ceil(row["sec"]), math.ceil(row["endSec"])
    #             phase = row["comment"]
                
    #             if phase not in phase_dict.values():
    #                 phase_dict = add_value_to_dict(phase_dict, phase)
                    
    #             extract_frames(root, c, start, end, phase_dict, phase)
    
    
    # # Save the dictionary to a JSON file
    # file_path = f'{root}/phase_dict.json'
    # with open(file_path, 'w') as file:
    #     json.dump(phase_dict, file, indent=4)
        
    imgs = []
    phases = []
    for case in tqdm.tqdm(os.listdir(f"{root}/frames")):
        for img in os.listdir(f"{root}/frames/{case}"):
            imgs.append(f"{root}/frames/{case}/{img}")

            # phase
            phs = img.split(".")[0].split("_")[-1]
            print("phs: ", phs)
            print(phase_dict.keys())
            assert str(phs) in phase_dict.keys()
            phases.append(phs)
            
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame({
        "images":imgs,
        "phases":phases
    }, columns=["images","phases"])
    
    # Save the DataFrame to a CSV file if needed
    df.to_csv(f"{root}/extracted_frames_with_phases.csv", index=False)
