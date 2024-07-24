import os
import math
import pandas as pd
import tqdm

def seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def extract_frames(root, case, start, end, category):
    # Convert start and end times to HH:MM:SS format
    start_hms = seconds_to_hms(start)
    end_hms = seconds_to_hms(end)

    print("start, end: ", start_hms, end_hms, category)
    
    # Ensure the frames directory exists
    frames_dir = f"{root}/frames/{case}"
    os.makedirs(frames_dir, exist_ok=True)

    if category == "Tonifying/Antibiotics":
        category = "TA"
    if category == "Lens Implantation":
        category = "LI"
    if category == "Lens positioning":
        category = "LP"
    
    # Create the ffmpeg command
    cmd = f"ffmpeg -ss {start_hms} -to {end_hms} \
            -i {root}/videos/{case}.mp4 \
            -vf 'fps=1' {frames_dir}/frame_%04d_{start}_{end}_{category}.jpg"
    
    # Execute the command
    os.system(cmd)
    
if __name__ == "__main__":
    root = "/data/shared/cataract-1K/phase_recognition"
    cases = os.listdir(f"{root}/annotations")[:3]
    
    for c in cases:
        if c != "SYNAPSE_METADATA_MANIFEST.tsv":
            f = pd.read_csv(f"{root}/annotations/{c}/{c}_annotations_phases.csv")
            for _, row in f.iterrows():
                start, end = math.ceil(row["sec"]), math.ceil(row["endSec"])
                category = row["comment"]
                extract_frames(root, c, start, end, category)

    # imgs = []
    # phases = []
    # for case in tqdm.tqdm(os.listdir(f"{root}/frames")):
    #     f = pd.read_csv(f"{root}/annotations/{case}/{case}_annotations_phases.csv")
    #     for img in os.listdir(f"{root}/frames/{case}"):
    #         imgs.append(f"{root}/frames/{case}/{img}")
            
    #         # timing of the frame
    #         _, _, ss, tt = img.split(".")[0].split("_")
    #         print(ss,tt)
    #         for _,i in f.iterrows():
    #             start, end = math.ceil(i["sec"]), math.ceil(i["endSec"])
    #             print(start, end)
    #             if ss == start and tt == end:
    #                 phases.append(i["comment"])
    #                 break

    # # Convert list of dictionaries to DataFrame
    # df = pd.DataFrame({
    #     "images":imgs,
    #     "phases":phases
    # }, columns=["images","phases"])
    
    # # Save the DataFrame to a CSV file if needed
    # df.to_csv(f"{root}/extracted_frames_with_phases.csv", index=False)
