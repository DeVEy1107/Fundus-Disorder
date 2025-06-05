import os
import json
from datetime import datetime

import cv2
import numpy as np
import pydicom




def read_dicom_image(file_path: str = None) -> np.ndarray:    
    try:
        dicom_data = pydicom.dcmread(file_path)
    except Exception as e:
        print(f"Error reading DICOM file {file_path}: {e}")
        return None
    
    return dicom_data.pixel_array


def read_json_file(file_path: str = None) -> dict:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return None
    
    return data




def convert_to_coco_format(
    input_data, 
    dataset_name: str = "Custom Dataset",
    description: str = "Converted dataset"
):

    coco_data = {
        "info": {
            "description": description,
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown License",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    category_id = 1
    annotation_id = 1
    classes_map = {
        "1": "large cupping",
        "2": "lattice",
        "3": "break",
        "4": "other",
        "5": "maculopathy",
        "6": "floater",
        "7": "laser",
        "8": "hemorrhage",
        "9": "wrong image",
        "10": "RD",
        "11": "Pigment"
    }
    classes_keys = list(classes_map.keys())

    for image_idx, item in enumerate(input_data):

        image_info = {
            "id": image_idx + 1,
            "width": item["imageWidth"],
            "height": item["imageHeight"],
            "file_name": item["imagePath"].replace(".dcm", ".tiff"),
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        }
        coco_data["images"].append(image_info)

        for sub_item in item["shapes"]:
            
            if sub_item["label"] not in classes_keys:
                category_info = {
                    "id": int(sub_item["label"]),
                    "name": classes_map[sub_item["label"]],
                    "supercategory": ""
                }
                coco_data["categories"].append(category_info)
                category_id += 1

            polygon = np.array(sub_item["points"]).round().astype(int)
            if polygon.shape[0] < 3:
                print(f"Warning: Polygon has less than 3 points, skipping annotation for {sub_item['label']}")
                continue

            x1, y1 = int(polygon[:, 0].min()), int(polygon[:, 1].min())
            x2, y2 = int(polygon[:, 0].max()), int(polygon[:, 1].max())
            bbox_width, bbox_height = x2 - x1, y2 - y1

            annotation_info = {
                "id": annotation_id,
                "image_id": image_idx + 1,
                "category_id": int(sub_item["label"]),
                "bbox": [x1, y1, bbox_width, bbox_height],
                "segmentation": [polygon.flatten().tolist()],
                "area": bbox_width * bbox_height,
                "iscrowd": 0
            }
            coco_data["annotations"].append(annotation_info)
            annotation_id += 1
            
    return coco_data




def test_io_functions():
    # Test reading a DICOM file
    dicom_file = "C:/Users/B093022035/Desktop/Fundus-Disorder/data/retinal-001/1.2.826.0.2.139953.1.2.51872.44012.57444.5.dcm"
    image = read_dicom_image(dicom_file)
    if image is not None:
        print(f"Successfully read DICOM image with shape: {image.shape}")
    
    # Test reading a JSON file
    json_file = "C:/Users/B093022035/Desktop/Fundus-Disorder/data/retinal-001/1.2.826.0.2.139953.1.2.51872.44012.57444.5.json"
    data = read_json_file(json_file)
    if data is not None:
        print(f"Successfully read JSON data: {data}")



def test_COCO_convert():
    
    root_dir = "C:/Users/B093022035/Desktop/Fundus-Disorder/data/retinal-001"
    ids = [f.split(".dcm")[0] for f in os.listdir(root_dir) if f.endswith('.dcm')]
    json_filenames = [os.path.join(root_dir, f"{id}.json") for id in ids]

    input_data = []
    for json_file in json_filenames:
        data = read_json_file(json_file)
        if data is not None:
            input_data.append(data)
    coco_data = convert_to_coco_format(input_data, dataset_name="Retinal Dataset", description="Fundus Disorder Dataset")
    coco_output_file = "coco_format.json"
    with open(coco_output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)


if __name__ == "__main__":
    # test_io_functions()
    test_COCO_convert()
    print("COCO conversion completed successfully.")