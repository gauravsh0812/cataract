import os
import cv2
import pandas as pd
import albumentations as A
from tqdm import tqdm

# Define the augmentation pipeline
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.5),
    A.ElasticTransform(p=0.5),
])

def augment_and_save(image, mask, save_dir_images, save_dir_masks, base_name, index):
    augmented = augmentation_pipeline(image=image, mask=mask)
    aug_image = augmented['image']
    aug_mask = augmented['mask']
    
    # Save augmented image and mask
    aug_image_path = os.path.join(save_dir_images, f"{base_name}_aug_{index}.png")
    aug_mask_path = os.path.join(save_dir_masks, f"{base_name}_aug_{index}.png")
    
    cv2.imwrite(aug_image_path, aug_image)
    cv2.imwrite(aug_mask_path, aug_mask)
    
    return aug_image_path, aug_mask_path

def augment_dataset(csv_file, save_dir_images, save_dir_masks, output_csv_file, num_augmentations=3):
    # Read the CSV file
    df = pd.read_csv(csv_file)[:100]
    
    # Create directories if they don't exist
    os.makedirs(save_dir_images, exist_ok=True)
    os.makedirs(save_dir_masks, exist_ok=True)
    
    # Prepare a new DataFrame to store augmented data
    augmented_data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        base_name = os.path.splitext(os.path.basename(row['Image_Paths']))[0]
        image_path = row['Image_Paths']
        mask_path = row['Mask_Paths']
        
        # Read the image and mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # mask is grayscale
        
        # Save original image and mask
        original_image_path = os.path.join(save_dir_images, f"{base_name}.png")
        original_mask_path = os.path.join(save_dir_masks, f"{base_name}.png")
        cv2.imwrite(original_image_path, image)
        cv2.imwrite(original_mask_path, mask)
        
        # Add the original paths and associated metadata to the new CSV
        augmented_data.append([original_image_path, original_mask_path, row['Labels'], row['Questions']])
        
        # Generate augmentations
        for i in range(num_augmentations):
            aug_image_path, aug_mask_path = augment_and_save(image, mask, save_dir_images, save_dir_masks, base_name, i + 1)
            # Add augmented paths and associated metadata to the new CSV
            augmented_data.append([aug_image_path, aug_mask_path, row['Labels'], row['Questions']])
    
    # Save the new CSV with augmented data
    augmented_df = pd.DataFrame(augmented_data, columns=['Image_Paths', 'Mask_Paths', 'Labels', 'Questions'])
    augmented_df.to_csv(output_csv_file, index=False)

# Directories
csv_file = "/data/shared/cataract-1K/segmentation/final_data_for_segmentation/final_dataset.csv"
save_dir_images = "/data/shared/cataract-1K/segmentation/final_data_for_segmentation/final_images"
save_dir_masks = "/data/shared/cataract-1K/segmentation/final_data_for_segmentation/masks"
output_csv_file = "/data/shared/cataract-1K/segmentation/final_data_for_segmentation/augmented_dataset.csv"

# Perform data augmentation
augment_dataset(csv_file, save_dir_images, save_dir_masks, output_csv_file, num_augmentations=3)  # Adjust num_augmentations as needed
