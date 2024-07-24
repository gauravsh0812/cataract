import os
import math
import pandas as pd

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

    # Create the ffmpeg command
    cmd = f"ffmpeg -ss {start_hms} -to {end_hms} \
            -i {root}/videos/{case}.mp4 \
            -vf 'fps=1' {frames_dir}/frame_%04d_{start}_{end}.jpg"
    
    # Execute the command
    os.system(cmd)
    
if __name__ == "__main__":
    root = "/data/shared/cataract-1K/phase_recognition"
    cases = os.listdir(f"{root}/annotations")

    tmp = dict()
    
    for c in cases:
        tmp[c] = []
        if c != "SYNAPSE_METADATA_MANIFEST.tsv":
            f = pd.read_csv(f"{root}/annotations/{c}/{c}_annotations_phases.csv")
            for _, row in f.iterrows():
                start, end = math.ceil(row["sec"]), math.ceil(row["endSec"])
                category = row["comment"]
                tmp[c].append([start,end,category])
                extract_frames(root, c, start, end, category)

    imgs = []
    phases = []
    for case in os.listdir(f"{root}/frames"):
        for img in os.listdir(case):
            imgs.append(f"{root}/frames/{case}/{img}")

            # timing of the frame
            _, _, ss, tt = img.split(".")[0].split("_")

            # get the phase
            for l in tmp[case]:
                s, t, ctgry = l
                if s == ss and t == tt:
                    phases.append(ctgry)
    
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame({
        "images":imgs,
        "phases":phases
    }, columns=["images","phases"])
    
    # Save the DataFrame to a CSV file if needed
    df.to_csv(f"{root}/extracted_frames_with_phases.csv", index=False)
