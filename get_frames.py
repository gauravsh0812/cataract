import os
import math
import pandas as pd

def seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def extract(root, case, start, end, category):
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
    
    # # List all frames generated during this extraction
    # frames = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_')])
    
    # # Get the corresponding frames within the time range and associate them with the category
    # extracted_frames = []
    # for frame in frames:
    #     frame_path = os.path.join(frames_dir, frame)
    #     extracted_frames.append({"start": start_hms, "end": end_hms,
    #                              "frame_path": frame_path, "phase": category})
    
    # return extracted_frames

if __name__ == "__main__":
    root = "/data/shared/cataract-1K/phase_recognition"
    cases = os.listdir(f"{root}/annotations")
    # all_frames = []
    
    for c in cases:
        if c != "SYNAPSE_METADATA_MANIFEST.tsv":
            f = pd.read_csv(f"{root}/annotations/{c}/{c}_annotations_phases.csv")
            for _, row in f.iterrows():
                start, end = math.ceil(row["sec"]), math.ceil(row["endSec"])
                category = row["comment"]
                # extracted_frames = extract(root, c, start, end, category)
                extract(root, c, start, end, category)
                # all_frames.extend(extracted_frames)
    
    # Convert list of dictionaries to DataFrame
    # df = pd.DataFrame(all_frames)
    
    # Save the DataFrame to a CSV file if needed
    # df.to_csv(f"{root}/extracted_frames_with_phases.csv", index=False)
