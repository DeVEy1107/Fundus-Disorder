import json
from datetime import datetime
from typing import List, Dict, Any

def map_label(original_label: str) -> int:
    """
    Map original labels to new label scheme:
    - Labels 2, 3 -> 1
    - Label 9 -> 2  
    - All other labels -> 3
    """
    try:
        label_int = int(original_label)
        if label_int in [2, 3]:
            return 1
        elif label_int == 9:
            return 2
        else:
            return 3
    except (ValueError, TypeError):
        # If label can't be converted to int, map to 3
        return 3

def clean_filename(filename: str) -> str:
    """Remove 'images/' prefix from filename if present."""
    if filename.startswith('images/'):
        return filename[7:]  # Remove 'images/' (7 characters)
    return filename

def convert_to_coco_format(input_data: List[Dict[str, Any]], 
                          dataset_name: str = "Custom Dataset",
                          description: str = "Converted dataset") -> Dict[str, Any]:
    """
    Convert custom annotation format to COCO JSON format.
    
    Args:
        input_data: List of image annotations in custom format
        dataset_name: Name of the dataset
        description: Description of the dataset
    
    Returns:
        Dictionary in COCO format
    """
    
    # Initialize COCO structure
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
    
    # Track unique categories/labels with mapped values
    category_map = {}
    category_id = 1
    annotation_id = 1
    
    # Process each image
    for image_idx, image_data in enumerate(input_data):
        # Clean filename by removing 'images/' prefix
        clean_file_name = clean_filename(image_data["image"])
        
        # Add image info
        image_info = {
            "id": image_idx + 1,
            "width": image_data["width"],
            "height": image_data["height"],
            "file_name": clean_file_name,
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        }
        coco_data["images"].append(image_info)
        
        # Process annotations for this image
        labels = image_data.get("labels", [])
        bboxes = image_data.get("bboxes", [])
        polygons = image_data.get("polygons", [])
        
        # Ensure all lists have the same length
        max_annotations = max(len(labels), len(bboxes), len(polygons))
        
        for ann_idx in range(max_annotations):
            # Get label and create category if needed
            if ann_idx < len(labels):
                original_label = labels[ann_idx]
                # Map the label according to the specified scheme
                mapped_label = map_label(original_label)
            else:
                # If no label available, skip this annotation
                continue
            
            # Check if polygon exists and is valid (if polygon validation is required)
            polygon_valid = True
            if ann_idx < len(polygons):
                polygon = polygons[ann_idx]
                polygon_valid = validate_polygon(polygon, image_data["width"], image_data["height"])
                
                if not polygon_valid:
                    print(f"Warning: Skipping annotation with original label '{original_label}' (mapped to {mapped_label}) due to invalid polygon in image {image_idx + 1}")
                    continue  # Skip this entire annotation
            
            # Create category if needed (only for valid annotations) using mapped label
            if mapped_label not in category_map:
                category_map[mapped_label] = category_id
                coco_data["categories"].append({
                    "id": category_id,
                    "name": str(mapped_label),
                    "supercategory": ""
                })
                category_id += 1
            current_category_id = category_map[mapped_label]
            
            annotation = {
                "id": annotation_id,
                "image_id": image_idx + 1,
                "category_id": current_category_id,
                "iscrowd": 0
            }
            
            # Add bounding box if available
            if ann_idx < len(bboxes):
                bbox = bboxes[ann_idx]  # [x_min, y_min, x_max, y_max]
                # Convert to COCO format: [x, y, width, height]
                x_min, y_min, x_max, y_max = bbox
                width = x_max - x_min
                height = y_max - y_min
                area = width * height
                
                annotation["bbox"] = [x_min, y_min, width, height]
                annotation["area"] = area
            
            # Add segmentation (polygon) - we know it's valid at this point
            if ann_idx < len(polygons):
                polygon = polygons[ann_idx]
                # Flatten polygon coordinates for COCO format
                segmentation = []
                for point in polygon:
                    segmentation.extend(point)  # [x1, y1, x2, y2, ...]
                
                annotation["segmentation"] = [segmentation]
                
                # Calculate area from polygon if bbox area not available
                if "area" not in annotation:
                    annotation["area"] = calculate_polygon_area(polygon)
            else:
                # If no polygon, set empty segmentation
                annotation["segmentation"] = []
            
            # Set default area if not calculated
            if "area" not in annotation:
                annotation["area"] = 0
            
            coco_data["annotations"].append(annotation)
            annotation_id += 1
    
    return coco_data

def validate_polygon(polygon: List[List[int]], image_width: int, image_height: int) -> bool:
    """
    Validate if a polygon is correct and usable.
    
    Args:
        polygon: List of [x, y] coordinates
        image_width: Width of the image
        image_height: Height of the image
    
    Returns:
        True if polygon is valid, False otherwise
    """
    if not polygon or len(polygon) < 3:
        return False
    
    # Check if all points are valid coordinates
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return False
        
        x, y = point
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return False
        
        # Check if coordinates are within image bounds
        if x < 0 or x >= image_width or y < 0 or y >= image_height:
            return False
    
    # Check if polygon is not degenerate (has area)
    area = calculate_polygon_area(polygon)
    if area < 1:  # Minimum area threshold
        return False
    
    # Check for self-intersection (basic check)
    if has_self_intersection(polygon):
        return False
    
    return True

def has_self_intersection(polygon: List[List[int]]) -> bool:
    """
    Basic check for self-intersecting polygons.
    
    Args:
        polygon: List of [x, y] coordinates
    
    Returns:
        True if polygon has self-intersection, False otherwise
    """
    n = len(polygon)
    if n < 4:
        return False
    
    # Check if any non-adjacent edges intersect
    for i in range(n):
        for j in range(i + 2, n):
            # Skip adjacent edges and last-first edge connection
            if (i == 0 and j == n - 1):
                continue
            
            if lines_intersect(polygon[i], polygon[(i + 1) % n], 
                             polygon[j], polygon[(j + 1) % n]):
                return True
    
    return False

def lines_intersect(p1: List[int], p2: List[int], p3: List[int], p4: List[int]) -> bool:
    """
    Check if two line segments intersect.
    
    Args:
        p1, p2: First line segment endpoints
        p3, p4: Second line segment endpoints
    
    Returns:
        True if segments intersect, False otherwise
    """
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def calculate_polygon_area(polygon: List[List[int]]) -> float:
    """
    Calculate area of a polygon using the shoelace formula.
    
    Args:
        polygon: List of [x, y] coordinates
    
    Returns:
        Area of the polygon
    """
    if len(polygon) < 3:
        return 0
    
    area = 0
    n = len(polygon)
    
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    
    return abs(area) / 2

def load_and_convert(input_file_path: str, output_file_path: str):
    """
    Load data from JSON file and convert to COCO format.
    
    Args:
        input_file_path: Path to input JSON file
        output_file_path: Path to save COCO format JSON
    """
    # Load input data
    with open(input_file_path, 'r') as f:
        input_data = json.load(f)
    
    # Convert to COCO format
    coco_data = convert_to_coco_format(input_data)
    
    # Save to file
    with open(output_file_path, 'w') as f:
        json.dump(coco_data, f, indent=4)
    
    print(f"Successfully converted {len(input_data)} images to COCO format")
    print(f"Output saved to: {output_file_path}")
    print(f"Categories found: {len(coco_data['categories'])}")
    print(f"Total valid annotations: {len(coco_data['annotations'])}")
    
    # Count annotations with polygons
    annotations_with_polygons = sum(1 for ann in coco_data['annotations'] if ann['segmentation'] and ann['segmentation'][0])
    print(f"Annotations with polygons: {annotations_with_polygons}")
    
    # Print label mapping summary
    print("\nLabel mapping summary:")
    print("Original labels -> Mapped labels:")
    print("2, 3 -> 1")
    print("9 -> 2") 
    print("Others -> 3")
    
    # Show actual categories found
    print(f"\nActual categories in dataset:")
    for category in coco_data['categories']:
        print(f"  Category ID: {category['id']}, Name: {category['name']}")

# Example usage
if __name__ == "__main__":
    # Example with your data structure
    sample_data = [
        {
            "id": "1.2.826.0.2.139953.1.2.51872.44012.57444.11",
            "width": 4000,
            "height": 4000,
            "color": "BGR",
            "image": "images/1.2.826.0.2.139953.1.2.51872.44012.57444.11.tiff",
            "labels": ["9", "2"],
            "bboxes": [
                [2161, 1822, 2400, 2170],
                [1732, 1547, 2625, 2386]
            ],
            "polygons": [
                [
                    [2161, 1890], [2237, 1822], [2287, 1831], [2350, 1886],
                    [2386, 1940], [2400, 2012], [2377, 2075], [2355, 2147],
                    [2269, 2170], [2237, 2143], [2206, 2111], [2183, 2039],
                    [2174, 1998], [2161, 1953]
                ],
                [
                    [1890, 1624], [1971, 1579], [2048, 1547], [2165, 1547],
                    [2260, 1561], [2440, 1619], [2508, 1737], [2576, 1872],
                    [2625, 2012], [2607, 2179], [2562, 2255], [2368, 2368],
                    [2188, 2386], [2012, 2355], [1931, 2251], [1859, 2183],
                    [1795, 2102], [1768, 2030], [1732, 1922], [1741, 1840],
                    [1804, 1764], [1845, 1683]
                ]
            ]
        }
    ]
    
    # Convert sample data
    coco_result = convert_to_coco_format(sample_data, "Sample Dataset", "Sample conversion with label mapping")
    
    # Print sample output
    print("Sample COCO format output:")
    print(json.dumps(coco_result, indent=2))
    
    # To use with your actual data file:
    # load_and_convert(r"C:\Users\B093022035\Desktop\Fundus-Disorder\data\retinal-tiff\annotations.json", "output_coco.json")