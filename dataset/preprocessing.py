import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.data import SequentialSampler
from box import Box
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)

# reading config file
with open("config.yaml") as f:
    cfg = Box(yaml.safe_load(f))

class Img2MML_dataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img = self.dataframe.iloc[index, 0] 
        lbl = self.dataframe.iloc[index,-1]
        return img,lbl

def preprocess(batch_size):

    print("creating dataloaders...")
    
    # two columns: [images, phases]
    if not cfg.dataset.load_image_tensors:
        df = pd.read_csv(f"{cfg.dataset.path_to_data}/extracted_frames_with_phases.csv")
    else:
        df = pd.read_csv(f"{cfg.dataset.path_to_data}/extracted_frames_tensors_with_phases.csv")

    for _ in range(5):
        df = df.sample(frac=1).reset_index(drop=True)

    # split the image_num into train, test, validate
    train_val_df, test_df = train_test_split(
        df, test_size=0.1, random_state=42
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.1, random_state=42
    )
    
    print(f"saving dataset files to {cfg.dataset.path_to_data}/ folder...")
    train_df.to_csv(f"{cfg.dataset.path_to_data}/train.csv", index=False)
    test_df.to_csv(f"{cfg.dataset.path_to_data}/test.csv", index=False)
    val_df.to_csv(f"{cfg.dataset.path_to_data}/val.csv", index=False)
    
    N = int(cfg.dataset.sample_size)
    if N != -1:
        train_df = train_df[:N]
        test_df = test_df[:int(N*0.1)]
        val_df = val_df[:int(N*0.1)]

    print("training dataset size: ", len(train_df))
    print("testing dataset size: ", len(test_df))
    print("validation dataset size: ", len(val_df))
    
    num_classes = cfg.dataset.num_classes

    # initailizing class Img2MML_dataset: train dataloader
    imml_train = Img2MML_dataset(train_df)
    # creating dataloader
    if cfg.general.ddp:
        train_sampler = DistributedSampler(
            dataset=imml_train,
            num_replicas=cfg.general.world_size,
            rank=cfg.general.rank,
            shuffle=cfg.dataset.shuffle,
        )
        sampler = train_sampler
        shuffle = False

    else:
        sampler = None
        shuffle = cfg.dataset.shuffle
        
    train_dataloader = DataLoader(
        imml_train,
        batch_size=batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=cfg.dataset.pin_memory,
    )

    # initailizing class Img2MML_dataset: val dataloader
    imml_val = Img2MML_dataset(val_df)
    if cfg.general.ddp:
        val_sampler = SequentialSampler(imml_val)
        sampler = val_sampler
        shuffle = False
    else:
        sampler = None
        shuffle = cfg.dataset.shuffle

    val_dataloader = DataLoader(
        imml_val,
        batch_size=batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=cfg.dataset.pin_memory,
    )

    # initailizing class Img2MML_dataset: test dataloader
    imml_test = Img2MML_dataset(test_df)
    if cfg.general.ddp:
        test_sampler = SequentialSampler(imml_test)
        sampler = test_sampler
        shuffle = False
    else:
        sampler = None
        shuffle = cfg.dataset.shuffle

    test_dataloader = DataLoader(
        imml_test,
        batch_size=batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=shuffle,
        sampler=None,
        pin_memory=cfg.dataset.pin_memory,
    )

    return (train_dataloader, 
            test_dataloader, 
            val_dataloader, num_classes)