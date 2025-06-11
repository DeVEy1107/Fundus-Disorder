import os
import torch
from datetime import datetime


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_dir, filename=None):
    """
    Save model checkpoint.
    
    Args:
        model: The model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        loss: Current loss
        save_dir: Directory to save checkpoint
        filename: Optional custom filename
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        filename = f'checkpoint_epoch_{epoch+1}.pth'
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    filepath = os.path.join(save_dir, filename)
    torch.save(checkpoint, filepath)
    print(f'Checkpoint saved: {filepath}')


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Starting epoch number
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f'Checkpoint loaded: epoch {epoch+1}, loss {loss:.4f}')
    return epoch + 1
