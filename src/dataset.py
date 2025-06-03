import os
import json

from typing import Tuple, Literal

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class RetinalDataset(Dataset):
    def __init__(
        self, 
        data_dir,
        target_ids = None
    ) -> None:
        
        self.data_dir = data_dir
        self.annotations = []

        with open(os.path.join(data_dir, 'annotations.json'), 'r') as f:
            annotations = json.load(f)

        for metadata in annotations:
            if metadata["id"] not in target_ids:
                continue

            img_path = os.path.join(data_dir, metadata['image'])
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file {img_path} does not exist.")            
            img_path = img_path.replace('\\', '/')
            metadata["image"] = img_path

            metadata["labels"] = self._map_labels(metadata["labels"])

            self.annotations.append(metadata)

    @staticmethod
    def _map_labels(labels):
        new_labels = []
        for label in labels:
            if label in ["2", "3"]:
                new_labels.append("1")
            elif label in ["9"]:
                new_labels.append("2")
            elif label in ["1", "4", "5", "6", "7", "8", "10", "11"]:
                new_labels.append("3")
            else:
                raise ValueError(f"Unknown label {label}.")
        return new_labels
    
    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx) -> Tuple[np.ndarray, dict]:

        metadata = self.annotations[idx]
        img_path = metadata['image']

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Image at {img_path} could not be read.")
        if len(img.shape) == 2:
            raise ValueError(f"Image at {img_path} is not a valid color img.")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img, metadata


def create_dataset(
    data_dir : str, 
    set_name : Literal['train', 'val', 'test'] = 'train'
) -> Dataset:
    
    with open(os.path.join(data_dir, "stratified_ids.json"), 'r') as f:
        stratified_ids_sets = json.load(f)
    target_ids = stratified_ids_sets[0][set_name]
    print(f"Creating {set_name} dataset with {len(target_ids)} target IDs.")
    return RetinalDataset(data_dir=data_dir, target_ids=target_ids)



if __name__ == "__main__":
    data_dir = "C:/Users/B093022035/Desktop/Fundus-Disorder/data/retinal-tiff"

    train_dataset = create_dataset(data_dir, set_name='train')
    val_dataset = create_dataset(data_dir, set_name='val')
    test_dataset = create_dataset(data_dir, set_name='test')

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    img, metadata = train_dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Metadata: {metadata}")