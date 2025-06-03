import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

from torchvision.models.detection import maskrcnn_resnet50_fpn



def create_mask_rcnn_model(num_classes, pretrained=False):
    """
    Create a Mask R-CNN model with a ResNet-50 backbone.

    Parameters
    ----------
    num_classes : int
        Number of classes in the dataset (including background).

    Returns
    -------
    torchvision.models.detection.MaskRCNN
        The Mask R-CNN model.
    """
    # Load a pre-trained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(weights=pretrained)

    # Replace the classifier with a new one for the specified number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model




if __name__ == "__main__":
    
    maskrcnn = create_mask_rcnn_model(num_classes=3)
    print(maskrcnn)

    total_params = sum(p.numel() for p in maskrcnn.parameters() if p.requires_grad)
    print(f"Total trainable parameters in Mask R-CNN: {total_params/1e6:.2f} M")

    # Chekc the input and output shapes of the model
    print(f"Input shape: {maskrcnn.backbone.body.conv1.weight.shape}")
    print(f"Output shape: {maskrcnn.roi_heads.box_predictor.bbox_pred.weight.shape}")

    maskrcnn.eval()  # Set the model to evaluation mode
    dummy_input = torch.randn(1, 3, 224, 224)  # Example input tensor
    dummy_output = maskrcnn(dummy_input)[0]
    # print(f"Dummy output: {dummy_output}")
    print(f"Key in dummy output: {dummy_output.keys()}")
    for key, value in dummy_output.items():
        print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else value}")

    
    
