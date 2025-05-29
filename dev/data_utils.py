import os
import json

import cv2
import numpy as np
import pydicom

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from utils import find_fundus_circle, crop_fundus_image




class FundusDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.dcm_prefixes = [f.split('.dcm')[0] for f in os.listdir(root_dir) if f.endswith('.dcm')]
        self.dcm_files = [os.path.join(root_dir, f"{prefix}.dcm") for prefix in self.dcm_prefixes]
        self.json_files = [os.path.join(root_dir, f"{prefix}.json") for prefix in self.dcm_prefixes]

    def __len__(self):
        return len(self.dcm_files)

    def __getitem__(self, index):

        dcm_file = os.path.join(self.root_dir, self.dcm_files[index])
        ds = pydicom.dcmread(dcm_file)
        img = ds.pixel_array

        json_file = os.path.join(self.root_dir, self.json_files[index])
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        img, label_data = self._processs_metadata(img, json_data)

        if self.transform:
            img = self.transform(img)

        return {
            'image': img,
            'metadata': label_data,
            'filename': self.dcm_files[index]
        }


    @staticmethod
    def _processs_metadata(img, json_data):
        """
        Process the image and metadata from the DICOM and JSON files.
        
        Parameters
        ----------
        img : numpy.ndarray
            The pixel array of the DICOM image.
        json_data : dict
            The metadata from the JSON file.

        Returns
        -------
        tuple
            Processed image, x, y, r coordinates of the fundus circle.
        """
        x, y, r = find_fundus_circle(img)
        img = crop_fundus_image(img, x, y, r)

        label_data = {}
        for data in json_data["shapes"]:
            label = data.get("label", f"No label found in {data}")
            points = data.get("points", f"No points found in {data}")
            points = np.array(points, dtype=np.float32)
            
            points[:, 0] = points[:, 0] - (x - r)
            points[:, 1] = points[:, 1] - (y - r)

            points = np.round(points).astype(np.int32)
            label_data[label] = points.tolist()

        return img, label_data




import cv2
import matplotlib.pyplot as plt


class FundusImageVisualizer:
    def __init__(self):
        pass        

    def visualize(self, metadata):
        img = metadata['image']
        labels = metadata['metadata']
        filename = metadata['filename']
        img = self._draw_labels(img, labels)
        plt.imshow(img)
        plt.title(f"Fundus Image: {filename}")
        plt.axis('off')
        plt.show()

    @staticmethod
    def _get_bbox(points):
        """
        Get the bounding box of a set of points.
        
        Parameters
        ----------
        points : numpy.ndarray
            Array of points in the format [[x1, y1], [x2, y2], ...].

        Returns
        -------
        tuple
            (x_min, y_min, x_max, y_max) coordinates of the bounding box.
        """
        x_min = np.min(points[:, 0])
        y_min = np.min(points[:, 1])
        x_max = np.max(points[:, 0])
        y_max = np.max(points[:, 1])
        return (x_min, y_min, x_max, y_max)

    def _draw_labels(self, img, labels):
        for label, points in labels.items():
            points = np.array(points, dtype=np.int32)
            bbox = self._get_bbox(points)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(255, 0, 0), thickness=2)
        return img


if __name__ == "__main__":
    # Example usage
    root_dir = r"C:\Users\B093022035\Desktop\Fundus-Disorder\data\fundus-images"
    dataset = FundusDataset(root_dir=root_dir)
    print(f"Found {len(dataset.dcm_files)} DICOM files and {len(dataset.json_files)} JSON files.")
    
    sample = dataset[0]
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Metadata: {sample['metadata']}")
    print(f"  Filename: {sample['filename']}")
    
    # Optionally visualize the image
    img = sample['image']

    vis = FundusImageVisualizer()
    vis.visualize(sample)
