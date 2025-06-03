import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional

class MaskRCNNLoss(nn.Module):
    """
    Loss function for Mask R-CNN combining:
    1. RPN classification and regression losses
    2. ROI head classification and regression losses  
    3. Mask segmentation loss
    """
    
    def __init__(self, 
                 box_fg_iou_thresh: float = 0.5,
                 box_bg_iou_thresh: float = 0.5,
                 box_batch_size_per_image: int = 512,
                 box_positive_fraction: float = 0.25,
                 bbox_loss_weight: float = 1.0,
                 mask_loss_weight: float = 1.0,
                 objectness_loss_weight: float = 1.0,
                 rpn_bbox_loss_weight: float = 1.0):
        
        super(MaskRCNNLoss, self).__init__()
        
        self.box_fg_iou_thresh = box_fg_iou_thresh
        self.box_bg_iou_thresh = box_bg_iou_thresh
        self.box_batch_size_per_image = box_batch_size_per_image
        self.box_positive_fraction = box_positive_fraction
        
        # Loss weights
        self.bbox_loss_weight = bbox_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.objectness_loss_weight = objectness_loss_weight
        self.rpn_bbox_loss_weight = rpn_bbox_loss_weight
        
    def forward(self, 
                predictions: Dict[str, Tensor], 
                targets: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """
        Args:
            predictions: Dict containing:
                - 'cls_logits': [N, num_classes] classification logits
                - 'bbox_regression': [N, num_classes*4] bbox regression
                - 'mask_logits': [N, num_classes, mask_height, mask_width] mask logits
                - 'objectness': [N, 1] RPN objectness scores
                - 'rpn_bbox_regression': [N, 4] RPN bbox regression
                
            targets: List of dicts, each containing:
                - 'boxes': [num_boxes, 4] ground truth boxes
                - 'labels': [num_boxes] ground truth class labels
                - 'masks': [num_boxes, height, width] ground truth masks
                
        Returns:
            Dict of losses
        """
        
        losses = {}
        
        # 1. Classification Loss (CrossEntropyLoss)
        if 'cls_logits' in predictions:
            cls_loss = self._classification_loss(
                predictions['cls_logits'], 
                targets
            )
            losses['loss_classifier'] = cls_loss * self.bbox_loss_weight
            
        # 2. Bounding Box Regression Loss (SmoothL1Loss)
        if 'bbox_regression' in predictions:
            bbox_loss = self._bbox_regression_loss(
                predictions['bbox_regression'],
                targets
            )
            losses['loss_box_reg'] = bbox_loss * self.bbox_loss_weight
            
        # 3. Mask Segmentation Loss (Binary CrossEntropyLoss)
        if 'mask_logits' in predictions:
            mask_loss = self._mask_loss(
                predictions['mask_logits'],
                targets
            )
            losses['loss_mask'] = mask_loss * self.mask_loss_weight
            
        # 4. RPN Objectness Loss
        if 'objectness' in predictions:
            objectness_loss = self._objectness_loss(
                predictions['objectness'],
                targets
            )
            losses['loss_objectness'] = objectness_loss * self.objectness_loss_weight
            
        # 5. RPN Bbox Regression Loss
        if 'rpn_bbox_regression' in predictions:
            rpn_bbox_loss = self._rpn_bbox_loss(
                predictions['rpn_bbox_regression'],
                targets
            )
            losses['loss_rpn_box_reg'] = rpn_bbox_loss * self.rpn_bbox_loss_weight
            
        return losses
    
    def _classification_loss(self, cls_logits: Tensor, targets: List[Dict]) -> Tensor:
        """Classification loss using CrossEntropyLoss"""
        # Extract labels from targets
        labels = []
        for target in targets:
            labels.append(target['labels'])
        
        if len(labels) == 0:
            return torch.tensor(0.0, device=cls_logits.device, requires_grad=True)
            
        labels = torch.cat(labels, dim=0)
        
        # Ensure labels are valid indices
        labels = torch.clamp(labels, 0, cls_logits.size(1) - 1)
        
        return F.cross_entropy(cls_logits, labels)
    
    def _bbox_regression_loss(self, bbox_pred: Tensor, targets: List[Dict]) -> Tensor:
        """Bounding box regression loss using SmoothL1Loss"""
        # Extract ground truth boxes and labels
        gt_boxes = []
        labels = []
        
        for target in targets:
            gt_boxes.append(target['boxes'])
            labels.append(target['labels'])
            
        if len(gt_boxes) == 0:
            return torch.tensor(0.0, device=bbox_pred.device, requires_grad=True)
            
        gt_boxes = torch.cat(gt_boxes, dim=0)
        labels = torch.cat(labels, dim=0)
        
        # Only compute loss for positive samples (non-background)
        pos_indices = labels > 0
        if pos_indices.sum() == 0:
            return torch.tensor(0.0, device=bbox_pred.device, requires_grad=True)
            
        # Select predictions for positive samples
        num_classes = bbox_pred.size(1) // 4
        bbox_pred = bbox_pred.view(-1, num_classes, 4)
        
        pos_bbox_pred = bbox_pred[pos_indices]
        pos_labels = labels[pos_indices]
        pos_gt_boxes = gt_boxes[pos_indices]
        
        # Select class-specific predictions
        bbox_pred_selected = pos_bbox_pred[torch.arange(len(pos_labels)), pos_labels]
        
        return F.smooth_l1_loss(bbox_pred_selected, pos_gt_boxes, reduction='mean')
    
    def _mask_loss(self, mask_logits: Tensor, targets: List[Dict]) -> Tensor:
        """Mask segmentation loss using binary cross entropy"""
        # Extract ground truth masks and labels
        gt_masks = []
        labels = []
        
        for target in targets:
            if 'masks' in target:
                gt_masks.append(target['masks'])
                labels.append(target['labels'])
                
        if len(gt_masks) == 0:
            return torch.tensor(0.0, device=mask_logits.device, requires_grad=True)
            
        gt_masks = torch.cat(gt_masks, dim=0)
        labels = torch.cat(labels, dim=0)
        
        # Only compute loss for positive samples
        pos_indices = labels > 0
        if pos_indices.sum() == 0:
            return torch.tensor(0.0, device=mask_logits.device, requires_grad=True)
            
        pos_mask_logits = mask_logits[pos_indices]
        pos_labels = labels[pos_indices]
        pos_gt_masks = gt_masks[pos_indices]
        
        # Resize ground truth masks to match prediction size
        mask_h, mask_w = pos_mask_logits.shape[-2:]
        pos_gt_masks = F.interpolate(
            pos_gt_masks.unsqueeze(1).float(),
            size=(mask_h, mask_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        # Select class-specific mask predictions
        mask_pred_selected = pos_mask_logits[torch.arange(len(pos_labels)), pos_labels]
        
        return F.binary_cross_entropy_with_logits(
            mask_pred_selected, pos_gt_masks, reduction='mean'
        )
    
    def _objectness_loss(self, objectness: Tensor, targets: List[Dict]) -> Tensor:
        """RPN objectness loss"""
        # This is a simplified version - in practice, you'd need to match
        # anchors to ground truth boxes to determine positive/negative samples
        
        # For demonstration, assuming binary labels are provided
        if len(targets) == 0:
            return torch.tensor(0.0, device=objectness.device, requires_grad=True)
            
        # Create dummy objectness targets (should be computed from anchor matching)
        objectness_targets = torch.zeros_like(objectness.squeeze(-1))
        
        # Set some targets to 1 for positive samples (simplified)
        if len(targets) > 0 and 'boxes' in targets[0]:
            num_pos = min(len(targets[0]['boxes']), len(objectness_targets))
            objectness_targets[:num_pos] = 1
            
        return F.binary_cross_entropy_with_logits(
            objectness.squeeze(-1), objectness_targets, reduction='mean'
        )
    
    def _rpn_bbox_loss(self, rpn_bbox_pred: Tensor, targets: List[Dict]) -> Tensor:
        """RPN bounding box regression loss"""
        # This is simplified - in practice, you'd compute regression targets
        # from anchor-to-ground-truth matching
        
        if len(targets) == 0:
            return torch.tensor(0.0, device=rpn_bbox_pred.device, requires_grad=True)
            
        # Create dummy regression targets (should be computed from anchor matching)
        regression_targets = torch.zeros_like(rpn_bbox_pred)
        
        return F.smooth_l1_loss(rpn_bbox_pred, regression_targets, reduction='mean')
