import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau
from typing import Dict, List, Optional, Union
import math

class MaskRCNNOptimizer:
    """
    Optimizer configuration for Mask R-CNN training with different learning rates
    for backbone, neck, RPN, ROI head, and mask head components.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 base_lr: float = 0.02,
                 backbone_lr_mult: float = 0.1,
                 neck_lr_mult: float = 1.0,
                 rpn_lr_mult: float = 1.0,
                 roi_head_lr_mult: float = 1.0,
                 mask_head_lr_mult: float = 1.0,
                 weight_decay: float = 0.0001,
                 momentum: float = 0.9,
                 optimizer_type: str = 'SGD',
                 scheduler_type: str = 'MultiStepLR',
                 warmup_epochs: int = 1,
                 total_epochs: int = 12):
        """
        Args:
            model: The Mask R-CNN model
            base_lr: Base learning rate
            backbone_lr_mult: Learning rate multiplier for backbone
            neck_lr_mult: Learning rate multiplier for neck (FPN)
            rpn_lr_mult: Learning rate multiplier for RPN
            roi_head_lr_mult: Learning rate multiplier for ROI head
            mask_head_lr_mult: Learning rate multiplier for mask head
            weight_decay: Weight decay for regularization
            momentum: Momentum for SGD optimizer
            optimizer_type: Type of optimizer ('SGD', 'Adam', 'AdamW')
            scheduler_type: Type of scheduler ('MultiStepLR', 'CosineAnnealing', 'ReduceLROnPlateau')
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
        """
        
        self.model = model
        self.base_lr = base_lr
        self.backbone_lr_mult = backbone_lr_mult
        self.neck_lr_mult = neck_lr_mult
        self.rpn_lr_mult = rpn_lr_mult
        self.roi_head_lr_mult = roi_head_lr_mult
        self.mask_head_lr_mult = mask_head_lr_mult
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        
        # Create parameter groups
        self.param_groups = self._create_parameter_groups()
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create scheduler
        self.scheduler = self._create_scheduler()
        
        # Warmup scheduler
        self.warmup_scheduler = self._create_warmup_scheduler()
        
        self.current_epoch = 0
        
    def _create_parameter_groups(self) -> List[Dict]:
        """Create parameter groups with different learning rates"""
        
        param_groups = []
        
        # Helper function to check if parameter should have weight decay
        def should_decay(name: str, param: torch.Tensor) -> bool:
            # Don't apply weight decay to bias terms and batch norm parameters
            if 'bias' in name:
                return False
            if 'bn' in name or 'norm' in name:
                return False
            if param.dim() <= 1:  # 1D parameters (bias, bn scale/shift)
                return False
            return True
        
        # Backbone parameters
        backbone_params = []
        backbone_params_no_decay = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                if should_decay(name, param):
                    backbone_params.append(param)
                else:
                    backbone_params_no_decay.append(param)
        
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': self.base_lr * self.backbone_lr_mult,
                'weight_decay': self.weight_decay,
                'name': 'backbone'
            })
            
        if backbone_params_no_decay:
            param_groups.append({
                'params': backbone_params_no_decay,
                'lr': self.base_lr * self.backbone_lr_mult,
                'weight_decay': 0.0,
                'name': 'backbone_no_decay'
            })
        
        # Neck (FPN) parameters
        neck_params = []
        neck_params_no_decay = []
        
        for name, param in self.model.named_parameters():
            if 'neck' in name or 'fpn' in name:
                if should_decay(name, param):
                    neck_params.append(param)
                else:
                    neck_params_no_decay.append(param)
        
        if neck_params:
            param_groups.append({
                'params': neck_params,
                'lr': self.base_lr * self.neck_lr_mult,
                'weight_decay': self.weight_decay,
                'name': 'neck'
            })
            
        if neck_params_no_decay:
            param_groups.append({
                'params': neck_params_no_decay,
                'lr': self.base_lr * self.neck_lr_mult,
                'weight_decay': 0.0,
                'name': 'neck_no_decay'
            })
        
        # RPN parameters
        rpn_params = []
        rpn_params_no_decay = []
        
        for name, param in self.model.named_parameters():
            if 'rpn' in name:
                if should_decay(name, param):
                    rpn_params.append(param)
                else:
                    rpn_params_no_decay.append(param)
        
        if rpn_params:
            param_groups.append({
                'params': rpn_params,
                'lr': self.base_lr * self.rpn_lr_mult,
                'weight_decay': self.weight_decay,
                'name': 'rpn'
            })
            
        if rpn_params_no_decay:
            param_groups.append({
                'params': rpn_params_no_decay,
                'lr': self.base_lr * self.rpn_lr_mult,
                'weight_decay': 0.0,
                'name': 'rpn_no_decay'
            })
        
        # ROI Head parameters
        roi_params = []
        roi_params_no_decay = []
        
        for name, param in self.model.named_parameters():
            if 'roi_head' in name or ('box_head' in name and 'mask' not in name):
                if should_decay(name, param):
                    roi_params.append(param)
                else:
                    roi_params_no_decay.append(param)
        
        if roi_params:
            param_groups.append({
                'params': roi_params,
                'lr': self.base_lr * self.roi_head_lr_mult,
                'weight_decay': self.weight_decay,
                'name': 'roi_head'
            })
            
        if roi_params_no_decay:
            param_groups.append({
                'params': roi_params_no_decay,
                'lr': self.base_lr * self.roi_head_lr_mult,
                'weight_decay': 0.0,
                'name': 'roi_head_no_decay'
            })
        
        # Mask Head parameters
        mask_params = []
        mask_params_no_decay = []
        
        for name, param in self.model.named_parameters():
            if 'mask_head' in name or 'mask' in name:
                if should_decay(name, param):
                    mask_params.append(param)
                else:
                    mask_params_no_decay.append(param)
        
        if mask_params:
            param_groups.append({
                'params': mask_params,
                'lr': self.base_lr * self.mask_head_lr_mult,
                'weight_decay': self.weight_decay,
                'name': 'mask_head'
            })
            
        if mask_params_no_decay:
            param_groups.append({
                'params': mask_params_no_decay,
                'lr': self.base_lr * self.mask_head_lr_mult,
                'weight_decay': 0.0,
                'name': 'mask_head_no_decay'
            })
        
        # Handle any remaining parameters
        remaining_params = []
        remaining_params_no_decay = []
        handled_params = set()
        
        for group in param_groups:
            for param in group['params']:
                handled_params.add(id(param))
        
        for name, param in self.model.named_parameters():
            if id(param) not in handled_params:
                if should_decay(name, param):
                    remaining_params.append(param)
                else:
                    remaining_params_no_decay.append(param)
        
        if remaining_params:
            param_groups.append({
                'params': remaining_params,
                'lr': self.base_lr,
                'weight_decay': self.weight_decay,
                'name': 'remaining'
            })
            
        if remaining_params_no_decay:
            param_groups.append({
                'params': remaining_params_no_decay,
                'lr': self.base_lr,
                'weight_decay': 0.0,
                'name': 'remaining_no_decay'
            })
        
        return param_groups
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer"""
        
        if self.optimizer_type.lower() == 'sgd':
            return optim.SGD(
                self.param_groups,
                momentum=self.momentum,
                nesterov=True
            )
        elif self.optimizer_type.lower() == 'adam':
            return optim.Adam(
                self.param_groups,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.optimizer_type.lower() == 'adamw':
            return optim.AdamW(
                self.param_groups,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create the learning rate scheduler"""
        
        if self.scheduler_type.lower() == 'multisteplr':
            # Typically decay at epochs 8 and 11 for 12-epoch training
            milestones = [int(0.67 * self.total_epochs), int(0.92 * self.total_epochs)]
            return MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
            
        elif self.scheduler_type.lower() == 'cosineannealinglr':
            return CosineAnnealingLR(
                self.optimizer, 
                T_max=self.total_epochs - self.warmup_epochs,
                eta_min=self.base_lr * 0.01
            )
            
        elif self.scheduler_type.lower() == 'reducelronplateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=2,
                verbose=True
            )
        elif self.scheduler_type.lower() == 'none':
            return None
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")
    
    def _create_warmup_scheduler(self) -> Optional['WarmupScheduler']:
        """Create warmup scheduler"""
        if self.warmup_epochs > 0:
            return WarmupScheduler(
                self.optimizer,
                warmup_epochs=self.warmup_epochs,
                base_lrs=[group['lr'] for group in self.param_groups]
            )
        return None
    
    def step(self, epoch: int, val_loss: Optional[float] = None):
        """Step the scheduler"""
        self.current_epoch = epoch
        
        if self.warmup_scheduler and epoch < self.warmup_epochs:
            self.warmup_scheduler.step(epoch)
        elif self.scheduler:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if val_loss is not None:
                    self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
    
    def get_lr(self) -> List[float]:
        """Get current learning rates"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def print_lr_info(self):
        """Print current learning rate information"""
        print("Current Learning Rates:")
        for i, group in enumerate(self.optimizer.param_groups):
            name = group.get('name', f'group_{i}')
            print(f"  {name}: {group['lr']:.6f}")


class WarmupScheduler:
    """Linear warmup scheduler"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int, base_lrs: List[float]):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lrs = base_lrs
        
    def step(self, epoch: int):
        """Step the warmup scheduler"""
        if epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = epoch / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * warmup_factor


# Utility functions for different training configurations
def get_coco_optimizer(model: nn.Module, total_epochs: int = 12) -> MaskRCNNOptimizer:
    """Get optimizer configuration for COCO training"""
    return MaskRCNNOptimizer(
        model=model,
        base_lr=0.02,
        backbone_lr_mult=0.1,  # Lower LR for pretrained backbone
        neck_lr_mult=1.0,
        rpn_lr_mult=1.0,
        roi_head_lr_mult=1.0,
        mask_head_lr_mult=1.0,
        weight_decay=0.0001,
        momentum=0.9,
        optimizer_type='SGD',
        scheduler_type='MultiStepLR',
        warmup_epochs=1,
        total_epochs=total_epochs
    )

def get_fine_tuning_optimizer(model: nn.Module, total_epochs: int = 24) -> MaskRCNNOptimizer:
    """Get optimizer configuration for fine-tuning on custom dataset"""
    return MaskRCNNOptimizer(
        model=model,
        base_lr=0.005,  # Lower base LR for fine-tuning
        backbone_lr_mult=0.01,  # Much lower LR for backbone
        neck_lr_mult=0.1,
        rpn_lr_mult=0.5,
        roi_head_lr_mult=1.0,
        mask_head_lr_mult=1.0,
        weight_decay=0.0001,
        momentum=0.9,
        optimizer_type='SGD',
        scheduler_type='CosineAnnealingLR',
        warmup_epochs=2,
        total_epochs=total_epochs
    )

def get_adamw_optimizer(model: nn.Module, total_epochs: int = 12) -> MaskRCNNOptimizer:
    """Get AdamW optimizer configuration"""
    return MaskRCNNOptimizer(
        model=model,
        base_lr=0.0001,
        backbone_lr_mult=0.1,
        neck_lr_mult=1.0,
        rpn_lr_mult=1.0,
        roi_head_lr_mult=1.0,
        mask_head_lr_mult=1.0,
        weight_decay=0.05,  # Higher weight decay for AdamW
        optimizer_type='AdamW',
        scheduler_type='CosineAnnealingLR',
        warmup_epochs=1,
        total_epochs=total_epochs
    )


# Example usage
def example_usage():
    """Example of how to use the MaskRCNN optimizer"""
    
    # Create a dummy model (in practice, use your actual Mask R-CNN model)
    class DummyMaskRCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            self.neck = nn.Conv2d(64, 256, 1)
            self.rpn = nn.Conv2d(256, 3, 1)
            self.roi_head = nn.Linear(256, 81)
            self.mask_head = nn.Conv2d(256, 80, 1)
    
    model = DummyMaskRCNN()
    
    # Create optimizer for COCO training
    optimizer_config = get_coco_optimizer(model, total_epochs=12)
    
    print("Optimizer created successfully!")
    optimizer_config.print_lr_info()
    
    # Simulate training loop
    for epoch in range(3):
        print(f"\nEpoch {epoch}:")
        
        # Training step (dummy)
        optimizer_config.optimizer.zero_grad()
        
        # Step scheduler
        optimizer_config.step(epoch)
        optimizer_config.print_lr_info()
    
    return optimizer_config

if __name__ == "__main__":
    example_usage()