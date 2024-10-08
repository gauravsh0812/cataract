import os
import json
import pandas as pd
import cv2
import numpy as np
import tqdm

imgs = []
masks = []
lbls = []
coords = []
qtns = []

def rename_lbl(l):
    
    dict_lbls = {'Spatula':"spatula", 
                'Phacoemulsification Tip':"phacoemulsification_tip", 
                'Pupil':"pupil", 
                'Cornea':"cornea", 
                'Gauge':"gauge", 
                'Lens':"lens", 
                'cornea1':"cornea", 
                'Capsulorhexis Cystotome':"capsulorhexis_cystotome", 
                'Irrigation-Aspiration':"irrigation_aspiration", 
                'pupil1':"pupil", 
                'Slit Knife':"slit_knife", 
                'Lens Injector':"lens_injector", 
                'Incision Knife':"incision_knife", 
                'Katena Forceps':"katena_forceps", 
                'Capsulorhexis Forceps':"capsulorhexis_forceps"}

    return dict_lbls[l]

def get_details(_path, iPath, mask_path):
    f = json.load(open(_path,"r"))
    obj = f["objects"]
    for o in range(len(obj)):
        lbl = rename_lbl(f["objects"][o]["classTitle"])
        points = f["objects"][o]["points"]
        exterior_coord = points["exterior"] 

        get_masks(exterior_coord, iPath, mask_path, lbl)

        imgs.append(iPath)
        lbls.append(lbl)
        coords.append(exterior_coord)

        if lbl in ["lens", "pupil", "cornea"]: 
            qtns.append(f"Segment the target area {lbl} in the image.")
        else:
            qtns.append(f"Segment the surgical instrument {lbl} in the image.")

def get_masks(coordinates, ipath, mpath, lbl):

    # Load the corresponding image
    image = cv2.imread(ipath)
    height, width = image.shape[:2]

    # Create an empty binary mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Convert the coordinates to a format suitable for cv2.fillPoly
    polygon = np.array([coordinates], dtype=np.int32)

    # Draw the filled polygon on the mask
    cv2.fillPoly(mask, polygon, 255)

    # Save the mask
    name = os.path.basename(ipath)
    mask_path = f"{mpath}/{lbl}_{name}"
    masks.append(mask_path)
    cv2.imwrite(mask_path, mask)

if __name__ == "__main__":
    root = "/data/shared/cataract-1K/segmentation/Annotations/Images-and-Supervisely-Annotations"
    os.makedirs("/data/shared/cataract-1K/segmentation/final_data_for_segmentation", 
            exist_ok=True)
    
    mask_path = "/data/shared/cataract-1K/segmentation/final_data_for_segmentation/masks"
    os.makedirs(mask_path, exist_ok=True)
    
    for folder in tqdm.tqdm(os.listdir(root)):
        if "case_" in folder:
            ann_path = os.path.join(root, f"{folder}/ann")
            img_path = os.path.join(root, f"{folder}/img")

            for i in os.listdir(img_path):
                if ".png" in i:
                    ann_file_path = os.path.join(ann_path, f"{i}.json")
                    img_file_path = os.path.join(img_path, i)
                    get_details(ann_file_path, img_file_path, mask_path)

    # df = pd.DataFrame({
    #     'Image_Paths': imgs,
    #     'Mask_Paths': masks,
    #     'Labels': lbls,
    #     'Coordinates': coords,
    #     'Questions': qtns,
    # })

    # df.to_csv("/data/shared/cataract-1K/segmentation/final_data_for_segmentation/final_dataset.csv")