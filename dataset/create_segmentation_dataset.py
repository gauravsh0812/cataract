import os
import json
import pandas as pd

root = "/data/shared/cataract-1K/segmentation/Annotations/Images-and-Supervisely-Annotations"
imgs = []
lbls = []
coords = []
qtns = []

def get_details_and_mask(_path, iPath):
    f = json.load(open(_path,"r"))
    obj = f["objects"]
    for o in range(len(obj)):
        lbl = f["objects"][o]["classTitle"]
        points = f["objects"][o]["points"]
        exterior_coord = points["exterior"] 

        imgs.append(iPath)
        lbls.append(lbl)
        coords.append(exterior_coord)

        if lbl in ["Lens", "Pupil", "Cornea", "cornea1", "pupil1"]: 
            qtns.append(f"Segment the target area {lbl} in the image.")
        else:
            qtns.append(f"Segment the surgical instrument {lbl} in the image.")

for folder in os.listdir(root):
    if "case_" in folder:
        ann_path = os.path.join(root, f"{folder}/ann")
        img_path = os.path.join(root, f"{folder}/img")

        for i in os.listdir(img_path):
            ann_file_path = os.path.join(ann_path, f"{i}.json")
            img_file_path = os.path.join(img_path, i)
            get_details_and_mask(ann_file_path, img_file_path)

df = pd.DataFrame({
    'Image_Paths': imgs,
    'Labels': lbls,
    'Coordinates': coords,
    'Questions': qtns,
})

df.to_csv("/data/shared/cataract-1K/segmentation/final_dataset_for_segmentation.csv")