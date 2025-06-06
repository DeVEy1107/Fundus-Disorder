import torch
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn


def get_faster_rcnn_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:

    if pretrained:
        model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    else:
        model = fasterrcnn_resnet50_fpn(weights=None)
    print(f"Using Faster R-CNN model with pretrained weights: {pretrained}")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_retinanet_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:

    if pretrained:
        model = retinanet_resnet50_fpn(weights='DEFAULT')
    else:
        model = retinanet_resnet50_fpn(weights=None)
    print(f"Using RetinaNet model with pretrained weights: {pretrained}")

    in_features = model.head.classification_head.conv[0].out_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = num_classes
    model.head.classification_head.cls_logits = torch.nn.Conv2d(
        in_features, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
    )

    return model
    



if __name__ == "__main__":
    # Example usage
    num_classes = 3
    # model = get_faster_rcnn_model(num_classes)
    model = get_retinanet_model(num_classes, False)
    # print(model)

    # dummy_input = torch.randn(1, 3, 224, 224)  # Example input tensor
    # model.eval()  # Set the model to evaluation mode
    # with torch.no_grad():
    #     output = model(dummy_input)  # Forward pass
    # print(f"Output: {output}")
    # boxes = output[0]['boxes']
    # labels = output[0]['labels']
    # scores = output[0]['scores']
    # print(f"Boxes: {boxes.shape}")
    # print(f"Labels: {labels.shape}")
    # print(f"Scores: {scores.shape}")

    targets = [
        {
            "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            "labels": torch.tensor([1]),
            "image_id": torch.tensor([1]),
            "area": torch.tensor([400.0]),
            "iscrowd": torch.tensor([0])
        },
        {
            "boxes": torch.tensor([[15.0, 25.0, 35.0, 45.0]]),
            "labels": torch.tensor([2]),
            "image_id": torch.tensor([2]),
            "area": torch.tensor([400.0]),
            "iscrowd": torch.tensor([0])
        }
    ]
    dummy_input = torch.randn(2, 3, 224, 224)  # Example input tensor
    model.eval()  # Set the model to evaluation mode
    # losses_dict = model(dummy_input, targets=targets)  # Forward pass with targets

    losses_dict = model(dummy_input)
    print(f"Size of bbox: {losses_dict[0]['labels'].shape}")
    # print(f"Losses: {losses_dict}")