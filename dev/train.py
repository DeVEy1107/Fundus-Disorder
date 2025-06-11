import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time

import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from dataset import RetinalDataset, collate_fn
from model import get_faster_rcnn_model, get_retinanet_model
from optim import get_adam, get_step_lr_scheduler
from torchvision.ops import box_iou

from utils import save_checkpoint, load_checkpoint
from evaluation import calculate_map
from visual import plot_training_curves


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """
    Train the model for one epoch.
    
    Args:
        model: The Faster R-CNN model
        optimizer: Optimizer
        data_loader: Training data loader
        device: Device to run on
        epoch: Current epoch number
        print_freq: Print frequency for logging
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = len(data_loader)
    
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
        if batch_idx % print_freq == 0:
            print(f'Epoch [{epoch+1}] Batch [{batch_idx}/{num_batches}] '
                  f'Loss: {losses.item():.4f} '
                  f'Avg Loss: {total_loss/(batch_idx+1):.4f}')
    
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, data_loader, device):
    """
    Evaluate the model on validation set.
    
    Args:
        model: The Faster R-CNN model
        data_loader: Validation data loader
        device: Device to run on
    
    Returns:
        Tuple of (average_loss, mAP)
    """
    
    model.eval()
    total_loss = 0.0
    num_batches = len(data_loader)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            
            model.eval()
            predictions = model(images)
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    avg_loss = total_loss / num_batches
    
    mAP = calculate_map(all_predictions, all_targets, iou_threshold=0.5)
    
    return avg_loss, mAP




def main():
    # Configuration
    config = {
        'data_dir': "C:/Users/B093022035/Desktop/Fundus-Disorder/data/retinal-tiff",
        'img_dir': "C:/Users/B093022035/Desktop/Fundus-Disorder/data/retinal-tiff/images",
        'train_json': "splits/train_fold_1.json",
        'val_json': "splits/val_fold_1.json", 
        'test_json': "splits/test_fold_1.json",
        'num_classes': 12,  # 11 classes + background
        'batch_size': 2,   # Reduced for TIFF images which are typically large
        'num_workers': 2,  # Reduced for stability
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 0.0005,
        'step_size': 15,   # LR decay every 15 epochs
        'gamma': 0.1,      # LR decay factor
        'img_size': (512, 512),  # Resize images for faster training
        'save_dir': 'checkpoints',
        'print_freq': 5,
        'save_freq': 10,   # Save checkpoint every 10 epochs
        'resume_checkpoint': None  # Path to checkpoint to resume from
    }
    
    print("=" * 60)
    print("FASTER R-CNN TRAINING PIPELINE")
    print("=" * 60)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create datasets
    print("\nCreating datasets...")
    try:
        train_dataset = RetinalDataset(
            json_file=config['train_json'],
            img_dir=config['img_dir'],
            load_masks=False,  # Faster R-CNN doesn't need masks for detection
            img_size=config['img_size']
        )
        
        val_dataset = RetinalDataset(
            json_file=config['val_json'],
            img_dir=config['img_dir'],
            load_masks=False,
            img_size=config['img_size']
        )
        
        test_dataset = RetinalDataset(
            json_file=config['test_json'],
            img_dir=config['img_dir'],
            load_masks=False,
            img_size=config['img_size']
        )
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        
    except Exception as e:
        print(f"Error creating datasets: {e}")
        print("Please check that the JSON files and image directory exist.")
        return
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    print(f"\nCreating Faster R-CNN model with {config['num_classes']} classes...")
    model = get_faster_rcnn_model(config['num_classes'])
    # model = get_retinanet_model(config['num_classes'])
    model.to(device)
    
    # Create optimizer and scheduler
    optimizer = get_adam(model, config['learning_rate'], config['weight_decay'])
    scheduler = get_step_lr_scheduler(optimizer, config['step_size'], config['gamma'])
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if config['resume_checkpoint'] and os.path.exists(config['resume_checkpoint']):
        print(f"\nResuming from checkpoint: {config['resume_checkpoint']}")
        start_epoch = load_checkpoint(model, optimizer, scheduler, config['resume_checkpoint'])
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {config['num_epochs']} epochs...")
    print("=" * 60)
    
    # Training loop
    for epoch in range(start_epoch, config['num_epochs']):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 40)
        
        # Training phase
        print("Training...")
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, config['print_freq']
        )
        
        # Validation phase
        print("Validating...")
        val_loss, mAP_results = evaluate(model, val_loader, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  mAP: {mAP_results:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Epoch Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, 
                config['save_dir'], 'best_model.pth'
            )
            print(f"  *** New best model saved! Val Loss: {val_loss:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % config['save_freq'] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, 
                config['save_dir']
            )
        
        print("-" * 40)
    
    # Save final model
    print("\nTraining completed!")
    save_checkpoint(
        model, optimizer, scheduler, config['num_epochs']-1, val_losses[-1], 
        config['save_dir'], 'final_model.pth'
    )
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, config['save_dir'])
    
    # Print training summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Val Loss: {val_losses[-1]:.4f}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Total Epochs: {config['num_epochs']}")
    print(f"Checkpoints saved in: {config['save_dir']}")
    print("=" * 60)
    

if __name__ == "__main__":
    main()