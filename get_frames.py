import os
import math
import pandas as pd

def extract(root, case, start, end):
  
  cmd = f'ffmpeg \
        -ss 00:{start}:{0} \
        -to 00:{end}:{0} \
        -i {root}/videos/{case}.mp4 \
        -vf "fps=1" \
        {root}/frames/{case}/frame_%04d.jpg'
  
  os.system(cmd)

if __name__ == "__main__":
  root = "/data/shared/cataract-1K/phase_recognition"
  cases = os.listdir(f"{root}/annotations")
  df = pd.DataFrame(columns=["frame_path", "phase"])
  
  for c in cases:
    f = pd.read_csv(f"{root}/annotations/{c}/{c}_annotations_phases.csv")
    for _f in f:
      start, end = math.ceil(_f["sec"]), math.ceil(_f["endSec"])
      cmts = _f["comments"]
      extract(root, c, start, end)
