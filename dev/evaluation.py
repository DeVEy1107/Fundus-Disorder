import torch
from torchvision.ops import box_iou


def calculate_map(predictions, targets, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP) for object detection.
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        iou_threshold: IoU threshold for positive detection
    
    Returns:
        mAP score
    """
    
    # conver predictions and targets to CPU tensors if they are not already
    predictions = [{k: v.cpu() for k, v in pred.items()} for pred in predictions]
    targets = [{k: v.cpu() for k, v in target.items()} for target in targets]

    # Get unique classes
    all_classes = set()
    for target in targets:
        all_classes.update(target['labels'].cpu().numpy())
    
    if len(all_classes) == 0:
        return 0.0
    
    class_aps = []
    
    for class_id in all_classes:
        # Collect all predictions and targets for this class
        class_predictions = []
        class_targets = []
        
        for pred, target in zip(predictions, targets):
            # Get predictions for this class
            class_mask = pred['labels'] == class_id
            if class_mask.sum() > 0:
                class_pred = {
                    'boxes': pred['boxes'][class_mask],
                    'scores': pred['scores'][class_mask]
                }
                class_predictions.append(class_pred)
            else:
                class_predictions.append({'boxes': torch.empty(0, 4), 'scores': torch.empty(0)})
            
            # Get targets for this class
            target_mask = target['labels'] == class_id
            if target_mask.sum() > 0:
                class_target = target['boxes'][target_mask]
                class_targets.append(class_target)
            else:
                class_targets.append(torch.empty(0, 4))
        
        # Calculate AP for this class
        ap = calculate_ap_for_class(class_predictions, class_targets, iou_threshold)
        class_aps.append(ap)
    
    # Return mean of all class APs
    return sum(class_aps) / len(class_aps) if class_aps else 0.0


def calculate_ap_for_class(predictions, targets, iou_threshold):
    """
    Calculate Average Precision for a single class.
    
    Args:
        predictions: List of predictions for this class
        targets: List of targets for this class
        iou_threshold: IoU threshold
    
    Returns:
        Average Precision for this class
    """
    
    # Collect all predictions with their scores
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_image_ids = []
    
    for img_id, pred in enumerate(predictions):
        if len(pred['boxes']) > 0:
            all_pred_boxes.append(pred['boxes'])
            all_pred_scores.append(pred['scores'])
            all_pred_image_ids.extend([img_id] * len(pred['boxes']))
    
    if len(all_pred_boxes) == 0:
        return 0.0

    # Concatenate all predictions
    all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
    all_pred_scores = torch.cat(all_pred_scores, dim=0)
    all_pred_image_ids = torch.tensor(all_pred_image_ids, dtype=torch.long)
    
    # Sort predictions by confidence score
    sorted_indices = torch.argsort(all_pred_scores, descending=True)
    all_pred_boxes = all_pred_boxes[sorted_indices]
    all_pred_scores = all_pred_scores[sorted_indices]
    all_pred_image_ids = all_pred_image_ids[sorted_indices]
    
    # Count total number of ground truth boxes
    total_gt = sum(len(target) for target in targets)
    if total_gt == 0:
        return 0.0
    
    # Track which ground truth boxes have been matched
    gt_matched = [torch.zeros(len(target), dtype=torch.bool) for target in targets]
    
    # Calculate precision and recall for each prediction
    tp = torch.zeros(len(all_pred_boxes))
    fp = torch.zeros(len(all_pred_boxes))
    
    for i, (pred_box, pred_score, img_id) in enumerate(zip(all_pred_boxes, all_pred_scores, all_pred_image_ids)):
        img_id = img_id.item()
        gt_boxes = targets[img_id]
        
        if len(gt_boxes) == 0:
            fp[i] = 1
            continue
        
        # Calculate IoU with all ground truth boxes in this image
        ious = box_iou(pred_box.unsqueeze(0), gt_boxes)
        max_iou, max_idx = torch.max(ious, dim=1)
        
        if max_iou >= iou_threshold:
            if not gt_matched[img_id][max_idx]:
                tp[i] = 1
                gt_matched[img_id][max_idx] = True
            else:
                fp[i] = 1  # Already matched
        else:
            fp[i] = 1
    
    # Compute precision and recall
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    recall = tp_cumsum / total_gt
    
    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in torch.arange(0, 1.1, 0.1):
        if torch.sum(recall >= t) == 0:
            p = 0
        else:
            p = torch.max(precision[recall >= t])
        ap += p / 11
    
    return ap.item()


if __name__ == "__main__":

    # Create dummy data for testing
    predictions = [
        {'boxes': torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]]), 'scores': torch.tensor([0.9, 0.8]), 'labels': torch.tensor([1, 2])},
        {'boxes': torch.tensor([[15, 15, 55, 55]]), 'scores': torch.tensor([0.85]), 'labels': torch.tensor([1])}
    ]

    targets = [
        {'boxes': torch.tensor([[10, 10, 50, 50], [30, 30, 70, 70]]), 'labels': torch.tensor([1, 2])},
        {'boxes': torch.tensor([[20, 20, 60, 60]]), 'labels': torch.tensor([1])}
    ]

    # Convert to CPU tensors if necessary
    predictions = [{k: v.cpu() for k, v in pred.items()} for pred in predictions]
    targets = [{k: v.cpu() for k, v in target.items()} for target in targets]

    iou_threshold = 0.5
    mAP = calculate_map(predictions, targets, iou_threshold)
    print(f"Mean Average Precision (mAP): {mAP:.4f}")