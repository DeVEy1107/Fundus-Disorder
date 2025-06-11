import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import torchvision.transforms as transforms
import cv2

from preprocess import apply_clahe  # Assuming this is defined in preprocess.py


class RetinalDataset(Dataset):
    """
    PyTorch Dataset for loading COCO format annotations.
    
    This dataset loads images and their corresponding annotations including
    bounding boxes, segmentation masks, and category labels.
    """
    
    def __init__(self, 
                 json_file: str, 
                 img_dir: str, 
                 transform: Optional[Any] = None,
                 target_transform: Optional[Any] = None,
                 load_masks: bool = True,
                 img_size: Optional[Tuple[int, int]] = None,
                 augmentation: bool = False,
                 clahe: bool = False):    
        """
        Initialize the COCO dataset.
        
        Args:
            json_file: Path to the COCO format JSON annotation file
            img_dir: Directory containing the images
            transform: Optional transform to be applied to images
            target_transform: Optional transform to be applied to targets
            load_masks: Whether to load segmentation masks
            img_size: Optional target image size (width, height) for resizing
        """
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.load_masks = load_masks
        self.img_size = img_size
        self.augmentation = augmentation
        self.clahe = clahe
        
        # Load COCO annotations
        with open(json_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create mappings for efficient lookup
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # Group annotations by image_id
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # Get list of image IDs that have annotations
        self.img_ids = list(self.img_to_anns.keys())
        
        print(f"Loaded dataset with {len(self.img_ids)} images and {len(self.coco_data['annotations'])} annotations")
        print(f"Categories: {list(self.categories.keys())}")
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.img_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, target) where target is a dict containing:
            - boxes: Tensor of shape (N, 4) containing bounding boxes in [x_min, y_min, x_max, y_max] format
            - labels: Tensor of shape (N,) containing category labels
            - masks: Tensor of shape (N, H, W) containing segmentation masks (if load_masks=True)
            - image_id: Image ID
            - area: Tensor of shape (N,) containing annotation areas
            - iscrowd: Tensor of shape (N,) containing crowd flags
        """
        img_id = self.img_ids[idx]
        
        # Load image
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Handle different image formats
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        if self.clahe:
            # Apply CLAHE if enabled
            image = apply_clahe(np.asarray(image))
            image = Image.fromarray(image)

        # Get original image size
        orig_w, orig_h = image.size
        
        # Resize image if target size is specified
        if self.img_size is not None:
            target_w, target_h = self.img_size
            image = image.resize((target_w, target_h), Image.BILINEAR)
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h
        else:
            scale_x = scale_y = 1.0
        
        # Get annotations for this image
        anns = self.img_to_anns.get(img_id, [])
        
        # Prepare target dictionary
        target = {}
        
        if len(anns) > 0:
            # Extract bounding boxes and convert to torch tensors
            boxes = []
            labels = []
            areas = []
            iscrowd = []
            masks = []
            
            for ann in anns:
                # Bounding box in COCO format: [x, y, width, height]
                bbox = ann['bbox']
                x, y, w, h = bbox
                
                # Apply scaling if image was resized
                x = x * scale_x
                y = y * scale_y
                w = w * scale_x
                h = h * scale_y
                
                # Skip invalid boxes
                if w <= 0 or h <= 0:
                    print(f"Warning: Skipping invalid box with w={w}, h={h}")
                    continue
                
                # Convert from COCO format [x, y, width, height] to PyTorch format [x_min, y_min, x_max, y_max]
                x_min = x
                y_min = y
                x_max = x + w
                y_max = y + h
                
                # Ensure coordinates are within image bounds
                x_min = max(0, min(x_min, target_w if self.img_size else orig_w))
                y_min = max(0, min(y_min, target_h if self.img_size else orig_h))
                x_max = max(x_min + 1, min(x_max, target_w if self.img_size else orig_w))
                y_max = max(y_min + 1, min(y_max, target_h if self.img_size else orig_h))
                
                # Final validation
                if x_max <= x_min or y_max <= y_min:
                    print(f"Warning: Skipping degenerate box [{x_min}, {y_min}, {x_max}, {y_max}]")
                    continue
                
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(ann['category_id'])
                areas.append(ann.get('area', (x_max - x_min) * (y_max - y_min)))
                iscrowd.append(ann.get('iscrowd', 0))
                
                # Process segmentation mask if requested
                if self.load_masks and 'segmentation' in ann and ann['segmentation']:
                    mask = self._create_mask_from_segmentation(
                        ann['segmentation'], 
                        int(orig_w), int(orig_h),
                        scale_x, scale_y
                    )
                    masks.append(mask)
                elif self.load_masks:
                    # Create empty mask if no segmentation available
                    final_h = int(orig_h * scale_y) if self.img_size else orig_h
                    final_w = int(orig_w * scale_x) if self.img_size else orig_w
                    masks.append(np.zeros((final_h, final_w), dtype=np.uint8))
            
            # Convert to tensors
            if boxes:
                target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
                target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
                target['area'] = torch.as_tensor(areas, dtype=torch.float32)
                target['iscrowd'] = torch.as_tensor(iscrowd, dtype=torch.int64)
                
                if self.load_masks and masks:
                    target['masks'] = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
                    
                # Validate bounding boxes
                self._validate_boxes(target['boxes'], img_path)
                    
            else:
                # Empty annotations
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['labels'] = torch.zeros(0, dtype=torch.int64)
                target['area'] = torch.zeros(0, dtype=torch.float32)
                target['iscrowd'] = torch.zeros(0, dtype=torch.int64)
                
                if self.load_masks:
                    final_h = int(orig_h * scale_y) if self.img_size else orig_h
                    final_w = int(orig_w * scale_x) if self.img_size else orig_w
                    target['masks'] = torch.zeros((0, final_h, final_w), dtype=torch.uint8)
        else:
            # No annotations for this image
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros(0, dtype=torch.int64)
            target['area'] = torch.zeros(0, dtype=torch.float32)
            target['iscrowd'] = torch.zeros(0, dtype=torch.int64)
            
            if self.load_masks:
                final_h = int(orig_h * scale_y) if self.img_size else orig_h
                final_w = int(orig_w * scale_x) if self.img_size else orig_w
                target['masks'] = torch.zeros((0, final_h, final_w), dtype=torch.uint8)
        
        target['image_id'] = torch.tensor([img_id])
        
        # Apply transforms
        # Apply color augmentation if enabled (only for color, not position)
        if self.augmentation:
            color_aug = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            image = color_aug(image)
        
        
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default: convert PIL to tensor
            image = transforms.ToTensor()(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target
    
    def _validate_boxes(self, boxes: torch.Tensor, img_path: str):
        """
        Validate that all bounding boxes have positive width and height.
        
        Args:
            boxes: Tensor of bounding boxes in [x_min, y_min, x_max, y_max] format
            img_path: Path to the image (for error reporting)
        """
        if len(boxes) == 0:
            return
            
        # Check for positive width and height
        widths = boxes[:, 2] - boxes[:, 0]  # x_max - x_min
        heights = boxes[:, 3] - boxes[:, 1]  # y_max - y_min
        
        invalid_width = widths <= 0
        invalid_height = heights <= 0
        
        if invalid_width.any() or invalid_height.any():
            print(f"Error in image: {img_path}")
            for i, box in enumerate(boxes):
                if invalid_width[i] or invalid_height[i]:
                    w = widths[i].item()
                    h = heights[i].item()
                    print(f"  Invalid box {i}: {box.tolist()} (width={w:.2f}, height={h:.2f})")
            raise ValueError("Found invalid bounding boxes with non-positive width or height")
    
    def _create_mask_from_segmentation(self, segmentation: List[List[float]], 
                                     orig_w: int, orig_h: int,
                                     scale_x: float, scale_y: float) -> np.ndarray:
        """
        Create a binary mask from segmentation data.
        
        Args:
            segmentation: List of polygon coordinates
            orig_w: Original image width
            orig_h: Original image height
            scale_x: Scaling factor for width
            scale_y: Scaling factor for height
            
        Returns:
            Binary mask as numpy array
        """
        # Create mask with final dimensions
        final_h = int(orig_h * scale_y)
        final_w = int(orig_w * scale_x)
        mask = np.zeros((final_h, final_w), dtype=np.uint8)
        
        for seg in segmentation:
            if len(seg) < 6:  # Need at least 3 points (6 coordinates)
                continue
            
            # Reshape flat coordinates to points
            points = np.array(seg, dtype=np.float32).reshape(-1, 2)
            
            # Apply scaling
            points[:, 0] *= scale_x  # x coordinates
            points[:, 1] *= scale_y  # y coordinates
            
            # Convert to integer coordinates
            points = points.astype(np.int32)
            
            # Create mask for this polygon
            cv2.fillPoly(mask, [points], 1)
        
        return mask
    
    def get_category_names(self) -> Dict[int, str]:
        """Get mapping from category ID to category name."""
        return {cat_id: cat['name'] for cat_id, cat in self.categories.items()}
    
    def get_image_info(self, idx: int) -> Dict[str, Any]:
        """Get image information for a given index."""
        img_id = self.img_ids[idx]
        return self.images[img_id]
    
    def get_annotations_for_image(self, idx: int) -> List[Dict[str, Any]]:
        """Get all annotations for a given image index."""
        img_id = self.img_ids[idx]
        return self.img_to_anns.get(img_id, [])

def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle variable number of annotations.
    """
    images, targets = tuple(zip(*batch))
    images = torch.stack(images, 0)
    return images, targets

def create_data_loaders(train_json: str, 
                       val_json: str, 
                       test_json: str,
                       img_dir: str,
                       batch_size: int = 4,
                       num_workers: int = 4,
                       img_size: Optional[Tuple[int, int]] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        train_json: Path to training annotations JSON
        val_json: Path to validation annotations JSON  
        test_json: Path to test annotations JSON
        img_dir: Directory containing images
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
        img_size: Optional target image size (width, height)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = RetinalDataset(train_json, img_dir, transform=train_transform, img_size=img_size)
    val_dataset = RetinalDataset(val_json, img_dir, transform=val_transform, img_size=img_size)
    test_dataset = RetinalDataset(test_json, img_dir, transform=val_transform, img_size=img_size)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    dataset = RetinalDataset(
        json_file="splits/test_fold_1.json",
        img_dir="data/retinal-tiff/images",
        img_size=(512, 512),  # Optional: resize images to 512x512,
        clahe=True,  # Apply CLAHE for contrast enhancement
    )
    

    # Show the first image and its annotations
    for i in range(3):
        image, target = dataset[i]
        print(f"Image {i} size: {image.size()}")
        print(f"Target for image {i}: {target}")
        
        # Convert tensor to PIL for visualization
        image_pil = transforms.ToPILImage()(image)
        image_pil.show()
        
        # Print bounding boxes and labels
        if 'boxes' in target:
            print(f"Boxes: {target['boxes']}")
            print(f"Labels: {target['labels']}")
        
        if 'masks' in target:
            print(f"Masks shape: {target['masks'].shape}")
        else:
            print("No masks available for this image.")