import os
import argparse
import pandas as pd
import numpy as np
import io
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Configuration
YOLO_CLASSES = {"digit": 0}  # Binary classification: 0=digit
FRAME_SIZE = {
    "width": 256,  # Default image width
    "height": 256,  # Default image height
}
VAL_SPLIT = 0.2  # 20% of training data for validation
RANDOM_SEED = 42  # For reproducible splits

def create_yolov8_dataset_folders(folder: str, yolo_dataset_path: str) -> None:
    """Create the YOLOv8 dataset folders"""
    os.makedirs(f"{yolo_dataset_path}/{folder}/images", exist_ok=True)
    os.makedirs(f"{yolo_dataset_path}/{folder}/labels", exist_ok=True)
    print(f"Created YOLOv8 {folder} folders")


def process_to_yolo(df: pd.DataFrame, output_dir: str, dataset_type: str) -> None:
    """
    Process dataframe and save images/labels in YOLO format using OpenCV
    
    Args:
        df: Input DataFrame containing image and digit data. Expected format:
            - Each row must contain:
              * 'image' dict with:
                - 'bytes': image bytes (required)
                - 'path': optional image path
              * 'digits' dict with (optional for unlabeled data):
                - 'bbox': list of bounding boxes [x_min, y_min, width, height]
        output_dir: Base output directory for YOLO dataset
        dataset_type: Subfolder name (train/valid/test)

    Example DataFrame row:
    {
        'image': {
            'bytes': b'\x89PNG...',  # actual image bytes
            'path': 'image_1.png'    # optional
        },
        'digits': {
            'bbox': [
                [10, 20, 30, 40],   # x_min, y_min, width, height
                [50, 60, 20, 30]     # another digit
            ]
        }
    }
    """
    # Validate input DataFrame structure
    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    assert 'image' in df.columns, "DataFrame must contain 'image' column"
    assert df['image'].apply(lambda x: isinstance(x, dict)).all(), "All 'image' entries must be dictionaries"
    assert df['image'].apply(lambda x: 'bytes' in x).any(), "At least some images must have 'bytes' data"
    
    if 'digits' in df.columns:
        assert df['digits'].apply(lambda x: isinstance(x, dict) or pd.isna(x)).all(), \
            "'digits' entries must be either dictionaries or NaN"
    # Create output directories
    img_dir = os.path.join(output_dir, dataset_type, 'images')
    label_dir = os.path.join(output_dir, dataset_type, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    # Stats tracking
    stats = {
        'processed': 0,
        'skipped_invalid_bbox': 0,
        'skipped_corrupt_image': 0,
        'skipped_missing_data': 0
    }

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_type}"):
        try:
            # Get base filename
            img_path = row['image'].get('path', f'image_{idx}.png')
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Skip if no image data
            if 'bytes' not in row['image'] or not row['image']['bytes']:
                stats['skipped_missing_data'] += 1
                continue
                
            # Get image dimensions using OpenCV
            try:
                img_array = np.frombuffer(row['image']['bytes'], dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("OpenCV failed to decode image")
                img_height, img_width = img.shape[:2]
            except Exception as img_error:
                print(f"Error reading image {base_name}: {str(img_error)}")
                stats['skipped_corrupt_image'] += 1
                continue
            
            # Process bounding boxes
            yolo_lines = []
            if 'digits' in row and 'bbox' in row['digits'] and len(row['digits']['bbox']) > 0:
                for bbox in row['digits']['bbox']:
                    try:
                        # Convert bbox to numpy array if it isn't already
                        bbox = np.array(bbox)
                        
                        # Validate bbox format
                        if bbox.size != 4:
                            print(f"Invalid bbox format in {base_name}: {bbox}")
                            stats['skipped_invalid_bbox'] += 1
                            continue
                            
                        x_min, y_min, width, height = bbox
                        
                        # Convert to YOLO format (normalized center coordinates)
                        x_center = (x_min + width/2) / img_width
                        y_center = (y_min + height/2) / img_height
                        norm_width = width / img_width
                        norm_height = height / img_height
                        
                        # Validate normalized coordinates using numpy's logical functions
                        coords = np.array([x_center, y_center, norm_width, norm_height])
                        if np.any(coords < 0) or np.any(coords > 1):
                            print(f"Invalid coordinates in {base_name}: {coords}")
                            stats['skipped_invalid_bbox'] += 1
                            continue
                            
                        yolo_line = f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                        yolo_lines.append(yolo_line)
                        
                    except Exception as bbox_error:
                        print(f"Error processing bbox in {base_name}: {str(bbox_error)}")
                        stats['skipped_invalid_bbox'] += 1
                        continue
            
            # Save label file if we have valid annotations
            if yolo_lines:
                label_path = os.path.join(label_dir, f"{base_name}.txt")
                try:
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(yolo_lines))
                except Exception as label_error:
                    print(f"Error writing label file {label_path}: {str(label_error)}")
                    continue
            
            # Save image file
            img_path = os.path.join(img_dir, f"{base_name}.png")
            try:
                success = cv2.imwrite(img_path, img)
                if not success:
                    raise ValueError("OpenCV failed to write image")
                stats['processed'] += 1
            except Exception as img_save_error:
                print(f"Error saving image {img_path}: {str(img_save_error)}")
                
        except Exception as row_error:
            print(f"Error processing row {idx}: {str(row_error)}")
            continue
    
    # Print summary
    print(f"\n{dataset_type} set processing summary:")
    print(f"- Successfully processed: {stats['processed']}")
    print(f"- Skipped (invalid bbox): {stats['skipped_invalid_bbox']}")
    print(f"- Skipped (corrupt image): {stats['skipped_corrupt_image']}")
    print(f"- Skipped (missing data): {stats['skipped_missing_data']}")
    
    if sum(v for k,v in stats.items() if k != 'processed') > 0:
        print("Warning: Some samples were skipped. Check logs for details.")
        
def create_yolov8_yaml_file(
    yolo_dataset_path: str, yaml_file_path: str, folders: list[str]
) -> None:
    """Create the YOLOv8 yaml file"""
    with open(yaml_file_path, 'w') as file:
        for folder in folders:
            file.write(f"{folder}: {os.path.abspath(yolo_dataset_path)}/{folder}/images\n")
        file.write(f"\nnc: {len(YOLO_CLASSES)}\n")
        file.write(f"\nnames: {list(YOLO_CLASSES.keys())}\n")
    print(f"Created YOLOv8 config file at {yaml_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SVHN parquet files to YOLOv8 format")
    parser.add_argument(
        "--test-parquet",
        type=str,
        default="../../data/SVHN/test.parquet",
        help="Path to test.parquet file",
    )
    parser.add_argument(
        "--train-parquet",
        type=str,
        default="../../data/SVHN/train.parquet",
        help="Path to train.parquet file",
    )
    parser.add_argument(
        "--yolo-dataset-path",
        type=str,
        default="../../data/YoloSVHN",
        help="Path to output YOLOv8 dataset",
    )
    parser.add_argument(
        "--yaml-file-path",
        type=str,
        default="./yolo_svhn.yaml",
        help="Path to output YOLOv8 yaml file",
    )
    args = parser.parse_args()

    # Create dataset structure
    folders = ["train", "val", "test"]
    for folder in folders:
        create_yolov8_dataset_folders(folder, args.yolo_dataset_path)

    # Load and split training data
    train_df = pd.read_parquet(args.train_parquet)
    train_df, valid_df = train_test_split(
        train_df, 
        test_size=VAL_SPLIT, 
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    # Process all datasets
    process_to_yolo(train_df, args.yolo_dataset_path, "train")
    process_to_yolo(valid_df, args.yolo_dataset_path, "val")
    process_to_yolo(pd.read_parquet(args.test_parquet), args.yolo_dataset_path, "test")

    # Create YAML config
    create_yolov8_yaml_file(args.yolo_dataset_path, args.yaml_file_path, folders)