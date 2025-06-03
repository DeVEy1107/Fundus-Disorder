
## Format of the label

A single sample label in the dataset is represented as a dictionary with the following structure:

```json
{
    "id": "unique_id",              # Unique identifier for the image
    "image": "path/to/image.tiff",  # Path to the image file
    "width": 4000,                   # Width of the image
    "height": 4000,                  # Height of the image
    "color": "BGR",                # Color mode of the image (e.g., RGB, BGR, Grayscale)
    "annotations": [                # List of annotations for the image
        {
            "bboxes": [ # List of bounding boxes for the object
                [x1, y1, x2, y2],
                [x1, y1, x2, y2],
                ...   
            ],
            "labels": [ # List of labels corresponding to the bounding boxes
                "label1",
                "label2",
                ...
            ],
            "polygons": [ # List of polygons for the object
                [[x1, y1], [x2, y2], ...],
                [[x1, y1], [x2, y2], ...],
                ...
            ],
        },
        ...
    ]
}
```