import argparse
import os
import sys
import json
import time
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from PIL import Image, ImageDraw, ImageFont

# Import from src directory
from dataset import RetinalDataset, collate_fn
from model import get_faster_rcnn_model, get_retinanet_model
# from loss import FasterRCNNLoss

class ModelEvaluator:
    """
    Class for evaluating Faster R-CNN model on test set and visualizing results.
    """
    
    def __init__(self, 
                 model_path: str,
                 test_json: str,
                 img_dir: str,
                 output_dir: str = "../results",
                 img_size: tuple = (512, 512),
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.5,
                 device: str = None):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model checkpoint
            test_json: Path to test set JSON file
            img_dir: Directory containing images
            output_dir: Directory to save results
            img_size: Image size for inference
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS threshold for duplicate removal
            device: Device to run inference on
        """
        self.model_path = model_path
        self.test_json = test_json
        self.img_dir = img_dir
        self.output_dir = output_dir
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model
        self.model = self._load_model()
        
        # Load dataset
        self.test_dataset = RetinalDataset(
            json_file=test_json,
            img_dir=img_dir,
            load_masks=False,
            img_size=img_size,
            clahe=True # TODO: Set to True if you want to apply CLAHE
        )
        
        # Class names mapping (based on your label mapping)
        self.class_names = {
            1: "large cupping",
            2: "lattice",
            3: "break",
            4: "other",
            5: "maculopathy",
            6: "floater",
            7: "laser",
            8: "hemorrhage",
            9: "wrong image",
            10: "RD",
            11: "Pigment"
        }
        
        # Colors for visualization
        self.colors = {
            1: (255, 0, 0),    # Red
            2: (0, 255, 0),    # Green  
            3: (0, 0, 255),    # Blue
            4: (255, 255, 0),  # Yellow
            5: (255, 0, 255),  # Magenta
            6: (0, 255, 255),  # Cyan
            7: (128, 0, 128),  # Purple
            8: (0, 128, 0),    # Dark Green
            9: (128, 128, 0),  # Olive
            10: (128, 0, 0),   # Maroon
            11: (0, 0, 128),   # Navy
            12: (192, 192, 192) # Silver
        }
        
        print(f"Loaded test dataset with {len(self.test_dataset)} images")
    
    def _load_model(self):
        """Load the trained model from checkpoint."""
        print(f"Loading model from: {self.model_path}")
        
        # Create model
        model = get_faster_rcnn_model(num_classes=12)  # 11 classes + background
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
        return model
    
    def evaluate_model(self, save_results: bool = True):
        """
        Evaluate the model on the test set.
        
        Args:
            save_results: Whether to save detailed results to files
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print("Evaluating model on test set...")
        
        # Create data loader
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,  # Process one image at a time for detailed analysis
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn
        )
        
        # Evaluation metrics
        total_images = 0
        total_detections = 0
        class_detections = {}
        confidence_scores = []
        
        # Store detailed results
        detailed_results = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(test_loader):
                total_images += 1
                
                # Move to device
                images = [img.to(self.device) for img in images]
                
                # Inference
                predictions = self.model(images)
                pred = predictions[0]  # Single image batch
                target = targets[0]
                
                # Filter predictions by confidence
                keep_indices = pred['scores'] >= self.confidence_threshold
                filtered_boxes = pred['boxes'][keep_indices]
                filtered_labels = pred['labels'][keep_indices]
                filtered_scores = pred['scores'][keep_indices]
                
                # Apply NMS
                if len(filtered_boxes) > 0:
                    nms_indices = self._apply_nms(filtered_boxes, filtered_scores, self.nms_threshold)
                    final_boxes = filtered_boxes[nms_indices]
                    final_labels = filtered_labels[nms_indices]
                    final_scores = filtered_scores[nms_indices]
                else:
                    final_boxes = filtered_boxes
                    final_labels = filtered_labels
                    final_scores = filtered_scores
                
                # Update statistics
                num_detections = len(final_boxes)
                total_detections += num_detections
                
                for label in final_labels:
                    if label.item() in class_detections:
                        class_detections[label.item()] += 1
                    else:
                        class_detections[label.item()] = 1
                
                confidence_scores.extend(final_scores.cpu().numpy())
                
                # Store detailed result
                img_info = self.test_dataset.get_image_info(batch_idx)
                result = {
                    'image_id': img_info['id'],
                    'image_name': img_info['file_name'],
                    'num_detections': num_detections,
                    'boxes': final_boxes.cpu().numpy().tolist(),
                    'labels': final_labels.cpu().numpy().tolist(),
                    'scores': final_scores.cpu().numpy().tolist(),
                    'ground_truth': {
                        'boxes': target['boxes'].numpy().tolist(),
                        'labels': target['labels'].numpy().tolist()
                    }
                }
                detailed_results.append(result)
                
                # Print progress
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(test_loader)} images")
        
        # Calculate metrics
        avg_detections_per_image = total_detections / total_images if total_images > 0 else 0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        metrics = {
            'total_images': total_images,
            'total_detections': total_detections,
            'avg_detections_per_image': avg_detections_per_image,
            'class_detections': class_detections,
            'avg_confidence': avg_confidence,
            'confidence_scores': confidence_scores,
            'detailed_results': detailed_results
        }
        
        # Save results
        if save_results:
            self._save_evaluation_results(metrics)
        
        return metrics
    
    def _apply_nms(self, boxes, scores, iou_threshold):
        """Apply Non-Maximum Suppression."""
        from torchvision.ops import nms
        return nms(boxes, scores, iou_threshold)
    
    def visualize_predictions(self, 
                            num_images: int = 20,
                            save_images: bool = True,
                            show_ground_truth: bool = True):
        """
        Visualize model predictions on test images.
        
        Args:
            num_images: Number of images to visualize
            save_images: Whether to save visualization images
            show_ground_truth: Whether to show ground truth boxes
        """
        print(f"Visualizing predictions on {num_images} images...")
        
        # Create visualization directory
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Create data loader
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn
        )
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(test_loader):
                if batch_idx >= num_images:
                    break
                
                # Move to device
                images = [img.to(self.device) for img in images]
                
                # Inference
                predictions = self.model(images)
                pred = predictions[0]
                target = targets[0]
                
                # Get original image
                img_info = self.test_dataset.get_image_info(batch_idx)
                original_img_path = os.path.join(self.img_dir, img_info['file_name'])
                original_img = Image.open(original_img_path).convert('RGB')
                
                # Create visualization
                fig = self._create_visualization(
                    original_img, pred, target, img_info, show_ground_truth
                )
                
                if save_images:
                    save_path = os.path.join(vis_dir, f"prediction_{batch_idx:03d}_{img_info['file_name'].replace('.tiff', '.png')}")
                    fig.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                else:
                    plt.show()
                
                print(f"Visualized image {batch_idx + 1}/{num_images}")
    
    def _create_visualization(self, original_img, prediction, target, img_info, show_ground_truth=True):
        """Create a visualization figure for a single image."""
        # Resize original image to match model input size if needed
        if self.img_size:
            display_img = original_img.resize(self.img_size, Image.BILINEAR)
        else:
            display_img = original_img
        
        # Convert to numpy for matplotlib
        img_array = np.array(display_img)
        
        # Create figure
        fig, axes = plt.subplots(1, 2 if show_ground_truth else 1, figsize=(15, 7))
        if not show_ground_truth:
            axes = [axes]
        
        # Filter predictions
        keep_indices = prediction['scores'] >= self.confidence_threshold
        pred_boxes = prediction['boxes'][keep_indices]
        pred_labels = prediction['labels'][keep_indices]
        pred_scores = prediction['scores'][keep_indices]
        
        # Apply NMS
        if len(pred_boxes) > 0:
            nms_indices = self._apply_nms(pred_boxes, pred_scores, self.nms_threshold)
            pred_boxes = pred_boxes[nms_indices]
            pred_labels = pred_labels[nms_indices]
            pred_scores = pred_scores[nms_indices]
        
        # Plot predictions
        axes[0].imshow(img_array)
        axes[0].set_title(f"Predictions (conf ≥ {self.confidence_threshold})\n{img_info['file_name']}")
        axes[0].axis('off')
        
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            x1, y1, x2, y2 = box.cpu().numpy()
            width = x2 - x1
            height = y2 - y1
            
            color = np.array(self.colors[label.item()]) / 255.0
            
            rect = Rectangle((x1, y1), width, height, 
                           linewidth=2, edgecolor=color, facecolor='none')
            axes[0].add_patch(rect)
            
            # Add label with confidence
            label_text = f"{self.class_names[label.item()]}: {score:.2f}"
            axes[0].text(x1, y1-5, label_text, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                        fontsize=8, color='white', weight='bold')
        
        # Plot ground truth
        if show_ground_truth:
            axes[1].imshow(img_array)
            axes[1].set_title(f"Ground Truth\n{len(target['labels'])} annotations")
            axes[1].axis('off')
            
            for box, label in zip(target['boxes'], target['labels']):
                x1, y1, x2, y2 = box.numpy()
                width = x2 - x1
                height = y2 - y1
                
                color = np.array(self.colors[label.item()]) / 255.0
                
                rect = Rectangle((x1, y1), width, height,
                               linewidth=2, edgecolor=color, facecolor='none', linestyle='--')
                axes[1].add_patch(rect)
                
                # Add label
                label_text = self.class_names[label.item()]
                axes[1].text(x1, y1-5, label_text,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                            fontsize=8, color='white', weight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_summary_plots(self, metrics):
        """Create summary plots for the evaluation results."""
        print("Creating summary plots...")
        
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Class distribution plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Class detections
        classes = list(metrics['class_detections'].keys())
        counts = list(metrics['class_detections'].values())
        class_labels = [self.class_names.get(c, f"Class_{c}") for c in classes]
        
        axes[0, 0].bar(class_labels, counts, color=[np.array(self.colors[c])/255.0 for c in classes])
        axes[0, 0].set_title('Detections per Class')
        axes[0, 0].set_ylabel('Number of Detections')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Confidence score distribution
        if metrics['confidence_scores']:
            axes[0, 1].hist(metrics['confidence_scores'], bins=20, alpha=0.7, color='skyblue')
            axes[0, 1].axvline(metrics['avg_confidence'], color='red', linestyle='--', 
                              label=f'Mean: {metrics["avg_confidence"]:.3f}')
            axes[0, 1].set_title('Confidence Score Distribution')
            axes[0, 1].set_xlabel('Confidence Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
        
        # Detections per image distribution
        detections_per_image = [len(result['boxes']) for result in metrics['detailed_results']]
        axes[1, 0].hist(detections_per_image, bins=max(1, max(detections_per_image)//2), 
                       alpha=0.7, color='lightgreen')
        axes[1, 0].axvline(metrics['avg_detections_per_image'], color='red', linestyle='--',
                          label=f'Mean: {metrics["avg_detections_per_image"]:.2f}')
        axes[1, 0].set_title('Detections per Image Distribution')
        axes[1, 0].set_xlabel('Number of Detections')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Summary statistics
        stats_text = f"""
        Total Images: {metrics['total_images']}
        Total Detections: {metrics['total_detections']}
        Avg Detections/Image: {metrics['avg_detections_per_image']:.2f}
        Avg Confidence: {metrics['avg_confidence']:.3f}
        
        Class Breakdown:
        • {self.class_names[1]}: {metrics['class_detections'].get(1, 0)}
        • {self.class_names[2]}: {metrics['class_detections'].get(2, 0)}
        • {self.class_names[3]}: {metrics['class_detections'].get(3, 0)}
        • {self.class_names[4]}: {metrics['class_detections'].get(4, 0)}
        • {self.class_names[5]}: {metrics['class_detections'].get(5, 0)}
        • {self.class_names[6]}: {metrics['class_detections'].get(6, 0)}
        • {self.class_names[7]}: {metrics['class_detections'].get(7, 0)}
        • {self.class_names[8]}: {metrics['class_detections'].get(8, 0)}
        • {self.class_names[9]}: {metrics['class_detections'].get(9, 0)}
        • {self.class_names[10]}: {metrics['class_detections'].get(10, 0)}
        • {self.class_names[11]}: {metrics['class_detections'].get(11, 0)}
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'evaluation_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary plots saved to: {plots_dir}")
    
    def _save_evaluation_results(self, metrics):
        """Save evaluation results to JSON file."""
        # Prepare data for JSON serialization
        json_metrics = {
            'total_images': metrics['total_images'],
            'total_detections': metrics['total_detections'],
            'avg_detections_per_image': metrics['avg_detections_per_image'],
            'class_detections': metrics['class_detections'],
            'avg_confidence': float(metrics['avg_confidence']),
            'confidence_scores': [float(x) for x in metrics['confidence_scores']],
            'detailed_results': metrics['detailed_results'],
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold
        }
        
        results_path = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"Detailed results saved to: {results_path}")

def main():
    # Configuration - Update these paths according to your setup
    config = {
        'model_path': './checkpoints/best_model.pth',  # Path to your trained model
        'test_json': './splits/test_fold_1.json',      # Test set JSON file
        'img_dir': './data/retinal-tiff/images',                    # Images directory
        'output_dir': './results',                     # Output directory
        'confidence_threshold': 0.5,                    # Minimum confidence for detections
        'nms_threshold': 0.5,                          # NMS threshold
        'num_vis_images': 40,                          # Number of images to visualize
        'img_size': (512, 512),                       # Image size for inference
        'device': None,                                # Device (None for auto-detect)
        'enable_visualization': True                   # Whether to create visualizations
    }
    
    print("=" * 60)
    print("FASTER R-CNN MODEL EVALUATION")
    print("=" * 60)
    print(f"Model: {config['model_path']}")
    print(f"Test set: {config['test_json']}")
    print(f"Image directory: {config['img_dir']}")
    print(f"Output directory: {config['output_dir']}")
    print(f"Confidence threshold: {config['confidence_threshold']}")
    print(f"NMS threshold: {config['nms_threshold']}")
    print("=" * 60)
    
    # Check if model file exists
    if not os.path.exists(config['model_path']):
        print(f"Error: Model file not found at {config['model_path']}")
        print("Please check the path or train a model first.")
        return
    
    # Check if test data exists
    if not os.path.exists(config['test_json']):
        print(f"Error: Test JSON file not found at {config['test_json']}")
        print("Please check the path or create the test splits first.")
        return
    
    if not os.path.exists(config['img_dir']):
        print(f"Error: Image directory not found at {config['img_dir']}")
        print("Please check the path.")
        return
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(
            model_path=config['model_path'],
            test_json=config['test_json'],
            img_dir=config['img_dir'],
            output_dir=config['output_dir'],
            img_size=config['img_size'],
            confidence_threshold=config['confidence_threshold'],
            nms_threshold=config['nms_threshold'],
            device=config['device']
        )
        
        # Evaluate model
        metrics = evaluator.evaluate_model()
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total images processed: {metrics['total_images']}")
        print(f"Total detections: {metrics['total_detections']}")
        print(f"Average detections per image: {metrics['avg_detections_per_image']:.2f}")
        print(f"Average confidence: {metrics['avg_confidence']:.3f}")
        print("\nClass-wise detections:")
        for class_id, count in metrics['class_detections'].items():
            class_name = evaluator.class_names.get(class_id, f"Class_{class_id}")
            print(f"  {class_name}: {count}")
        print("=" * 60)
        
        # Create summary plots
        evaluator.create_summary_plots(metrics)
        
        # Visualize predictions
        if config['enable_visualization']:
            evaluator.visualize_predictions(
                num_images=config['num_vis_images'],
                save_images=True,
                show_ground_truth=True
            )
        
        print(f"\nEvaluation complete! Results saved to: {config['output_dir']}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

