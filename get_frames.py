import os
import math
import pandas as pd

def extract(root, case, start, end, category):
    # Ensure the frames directory exists
    frames_dir = f"{root}/frames/{case}"
    os.makedirs(frames_dir, exist_ok=True)

    # Create the ffmpeg command
    cmd = f'ffmpeg -ss 00:{start}:00 -to 00:{end}:00 -i {root}/videos/{case}.mp4 -vf "fps=1" {frames_dir}/frame_%04d.jpg'
    
    # Execute the command
    os.system(cmd)
    
    # List all frames generated during this extraction
    frames = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_')])
    
    # Get the corresponding frames within the time range and associate them with the category
    extracted_frames = []
    for frame in frames:
        frame_path = os.path.join(frames_dir, frame)
        extracted_frames.append((frame_path, category))
    
    return extracted_frames

if __name__ == "__main__":
    root = "/data/shared/cataract-1K/phase_recognition"
    cases = os.listdir(f"{root}/annotations")
    df = pd.DataFrame(columns=["frame_path", "phase"])
    
    for c in cases:
        f = pd.read_csv(f"{root}/annotations/{c}/{c}_annotations_phases.csv")
        for _, row in f.iterrows():
            start, end = math.ceil(row["sec"]), math.ceil(row["endSec"])
            category = row["comments"]
            extracted_frames = extract(root, c, start, end, category)
            
            # Append extracted frames and their categories to the DataFrame
            for frame_path, phase in extracted_frames:
                df = df.append({"frame_path": frame_path, "phase": phase}, ignore_index=True)
    
    # Save the DataFrame to a CSV file if needed
    df.to_csv(f"{root}/extracted_frames_with_phases.csv", index=False)
