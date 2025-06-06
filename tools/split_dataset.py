import json
import os
from typing import List, Dict, Any
from data_utils import convert_to_coco_format

def load_original_annotations(root_dir: str) -> List[Dict[str, Any]]:
    """Load the original annotations JSON file."""
    ids = [f.split(".dcm")[0] for f in os.listdir(root_dir) if f.endswith('.dcm')]
    json_filenames = [os.path.join(root_dir, f"{id}.json") for id in ids]

    input_data = []
    for json_file in json_filenames:
        with open(json_file, 'r') as f:
            data = json.load(f)
        if data is not None:
            input_data.append(data)
    return input_data

def load_stratified_folds(folds_path: str) -> List[Dict[str, Any]]:
    """Load the stratified folds JSON file."""
    with open(folds_path, 'r') as f:
        return json.load(f)

def filter_annotations_by_ids(annotations: List[Dict[str, Any]], 
                             image_ids: List[str]) -> List[Dict[str, Any]]:
    """Filter annotations to only include specified image IDs."""
    id_set = set(image_ids)
    return [ann for ann in annotations if ann['imagePath'].split(".dcm")[0] in id_set]

def create_split_files(annotations_path: str, 
                      folds_path: str, 
                      output_dir: str = "splits",
                      fold_number: int = 1):
    """
    Create train, test, and validation JSON files for a specific fold.
    
    Args:
        annotations_path: Path to original annotations JSON file
        folds_path: Path to stratified folds JSON file
        output_dir: Directory to save the split files
        fold_number: Which fold to use (1-5)
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading original annotations...")
    original_annotations = load_original_annotations(annotations_path)
    print(f"Loaded {len(original_annotations)} original annotations")
    
    print("Loading stratified folds...")
    folds_data = load_stratified_folds(folds_path)
    
    # Find the specified fold
    target_fold = None
    for fold in folds_data:
        if fold['fold'] == fold_number:
            target_fold = fold
            break
    
    if target_fold is None:
        raise ValueError(f"Fold {fold_number} not found in the folds data")
    
    print(f"Using fold {fold_number}")
    print(f"Train IDs: {len(target_fold['train'])}")
    print(f"Validation IDs: {len(target_fold['val'])}")
    print(f"Test IDs: {len(target_fold['test'])}")
    
    # Filter annotations for each split
    print("\nFiltering annotations for each split...")
    
    train_annotations = filter_annotations_by_ids(original_annotations, target_fold['train'])
    val_annotations = filter_annotations_by_ids(original_annotations, target_fold['val'])
    test_annotations = filter_annotations_by_ids(original_annotations, target_fold['test'])
    
    print(f"Train annotations: {len(train_annotations)}")
    print(f"Validation annotations: {len(val_annotations)}")
    print(f"Test annotations: {len(test_annotations)}")
    
    # Convert each split to COCO format
    print("\nConverting to COCO format...")

    train_coco = convert_to_coco_format(
        train_annotations, 
        f"Training split from fold {fold_number}"
    )
    
    val_coco = convert_to_coco_format(
        val_annotations, 
        f"Validation split from fold {fold_number}"
    )
    
    test_coco = convert_to_coco_format(
        test_annotations, 
        f"Test split from fold {fold_number}"
    )
    
    # Save the split files
    print("\nSaving split files...")
    
    train_path = os.path.join(output_dir, f"train_fold_{fold_number}.json")
    val_path = os.path.join(output_dir, f"val_fold_{fold_number}.json")
    test_path = os.path.join(output_dir, f"test_fold_{fold_number}.json")
    
    with open(train_path, 'w') as f:
        json.dump(train_coco, f, indent=2)
    print(f"Saved training data to: {train_path}")
    
    with open(val_path, 'w') as f:
        json.dump(val_coco, f, indent=2)
    print(f"Saved validation data to: {val_path}")
    
    with open(test_path, 'w') as f:
        json.dump(test_coco, f, indent=2)
    print(f"Saved test data to: {test_path}")
    
    # Print summary statistics
    print(f"\n=== Summary for Fold {fold_number} ===")
    print(f"Training set:")
    print(f"  - Images: {len(train_coco['images'])}")
    print(f"  - Annotations: {len(train_coco['annotations'])}")
    print(f"  - Categories: {len(train_coco['categories'])}")
    
    print(f"Validation set:")
    print(f"  - Images: {len(val_coco['images'])}")
    print(f"  - Annotations: {len(val_coco['annotations'])}")
    print(f"  - Categories: {len(val_coco['categories'])}")
    
    print(f"Test set:")
    print(f"  - Images: {len(test_coco['images'])}")
    print(f"  - Annotations: {len(test_coco['annotations'])}")
    print(f"  - Categories: {len(test_coco['categories'])}")
    
    # Count annotations with polygons for each split
    train_polygons = sum(1 for ann in train_coco['annotations'] 
                        if ann['segmentation'] and ann['segmentation'][0])
    val_polygons = sum(1 for ann in val_coco['annotations'] 
                      if ann['segmentation'] and ann['segmentation'][0])
    test_polygons = sum(1 for ann in test_coco['annotations'] 
                       if ann['segmentation'] and ann['segmentation'][0])
    
    print(f"\nAnnotations with polygons:")
    print(f"  - Training: {train_polygons}")
    print(f"  - Validation: {val_polygons}")
    print(f"  - Test: {test_polygons}")

def create_all_folds(annotations_path: str, 
                    folds_path: str, 
                    output_dir: str = "splits"):
    """Create train/test/val splits for all 5 folds."""
    
    print("Creating splits for all 5 folds...\n")
    
    for fold_num in range(1, 6):
        print(f"\n{'='*50}")
        print(f"Processing Fold {fold_num}")
        print(f"{'='*50}")
        
        try:
            create_split_files(annotations_path, folds_path, output_dir, fold_num)
        except Exception as e:
            print(f"Error processing fold {fold_num}: {str(e)}")
            continue
    
    print(f"\n{'='*50}")
    print("All folds processed successfully!")
    print(f"Output files saved in: {output_dir}")
    print(f"{'='*50}")

def verify_splits(output_dir: str = "splits", fold_number: int = 1):
    """Verify that the splits don't have overlapping image IDs."""
    
    train_path = os.path.join(output_dir, f"train_fold_{fold_number}.json")
    val_path = os.path.join(output_dir, f"val_fold_{fold_number}.json")
    test_path = os.path.join(output_dir, f"test_fold_{fold_number}.json")
    
    # Load the splits
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(val_path, 'r') as f:
        val_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    # Extract image file names (which contain the IDs)
    train_images = {img['file_name'] for img in train_data['images']}
    val_images = {img['file_name'] for img in val_data['images']}
    test_images = {img['file_name'] for img in test_data['images']}
    
    # Check for overlaps
    train_val_overlap = train_images & val_images
    train_test_overlap = train_images & test_images
    val_test_overlap = val_images & test_images
    
    print(f"\n=== Verification for Fold {fold_number} ===")
    print(f"Train-Validation overlap: {len(train_val_overlap)} images")
    print(f"Train-Test overlap: {len(train_test_overlap)} images")
    print(f"Validation-Test overlap: {len(val_test_overlap)} images")
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("WARNING: Found overlapping images between splits!")
        if train_val_overlap:
            print(f"Train-Val overlap: {list(train_val_overlap)[:5]}...")
        if train_test_overlap:
            print(f"Train-Test overlap: {list(train_test_overlap)[:5]}...")
        if val_test_overlap:
            print(f"Val-Test overlap: {list(val_test_overlap)[:5]}...")
    else:
        print("✓ No overlapping images found - splits are clean!")

if __name__ == "__main__":
    # Configuration
    ANNOTATIONS_PATH = "data/retinal-combined"  # Path to your original annotations file
    FOLDS_PATH = "data/retinal-tiff/stratified_ids.json"     # Path to your stratified folds file
    OUTPUT_DIR = "splits2"                  # Directory to save split files
    
    # Example usage - create splits for fold 1
    print("Creating splits for fold 1...")
    create_split_files(ANNOTATIONS_PATH, FOLDS_PATH, OUTPUT_DIR, fold_number=1)
    
    # Verify the splits
    verify_splits(OUTPUT_DIR, fold_number=1)
    
    # Uncomment the line below to create splits for all 5 folds
    # create_all_folds(ANNOTATIONS_PATH, FOLDS_PATH, OUTPUT_DIR)
