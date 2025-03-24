import os
import json
import shutil
import argparse
import pandas as pd
import io
from tqdm import tqdm  # type: ignore

YOLO_CLASSES = {"digit": 1}  # Binary classification: 1=digit
FRAME_SIZE = {
    "width": 256,  # Default image width
    "height": 256,  # Default image height
}

def create_yolov8_dataset_folders(folder: str, yolo_dataset_path: str) -> None:
    """Create the YOLOv8 dataset folders"""
    os.makedirs(f"{yolo_dataset_path}/{folder}/images", exist_ok=True)
    os.makedirs(f"{yolo_dataset_path}/{folder}/labels", exist_ok=True)
    print(f"Done: the YOLOv8 dataset {folder} folders were successfully created.")

def process_parquet_to_yolo(parquet_path: str, folder: str, yolo_dataset_path: str) -> None:
    """
    Process parquet file and convert to YOLO format
    Args:
        parquet_path: Path to the parquet file
        folder: Dataset folder name (train/test/val)
        yolo_dataset_path: Path to output YOLO dataset
    """
    df = pd.read_parquet(parquet_path)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {folder}"):
        # Get image path (if available) or create one
        image_path = row['image'].get('path', f'image_{idx}.png')
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Get image dimensions
        try:
            img = Image.open(io.BytesIO(row['image']['bytes']))
            img_width, img_height = img.size
        except:
            img_width, img_height = FRAME_SIZE['width'], FRAME_SIZE['height']
        
        # Prepare YOLO content
        yolo_lines = []
        bboxes = row['digits']['bbox']
        
        # Check if there are any digits in this image
        if len(bboxes) > 0:
            for bbox in bboxes:
                x_min, y_min, width, height = bbox
                x_center = (x_min + width/2) / img_width
                y_center = (y_min + height/2) / img_height
                norm_width = width / img_width
                norm_height = height / img_height
                
                yolo_line = f"1 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                yolo_lines.append(yolo_line)
        
        # Save annotation file if there are digits
        if yolo_lines:
            txt_path = f"{yolo_dataset_path}/{folder}/labels/{base_name}.txt"
            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
        
        # Save image
        img_path = f"{yolo_dataset_path}/{folder}/images/{base_name}.png"
        with open(img_path, 'wb') as f:
            f.write(row['image']['bytes'])

def create_yolov8_yaml_file(
    yolo_dataset_path: str, yaml_file_path: str, folders: list[str]
) -> None:
    """Create the YOLOv8 yaml file"""
    with open(yaml_file_path, 'w') as file:
        for folder in folders:
            file.write(f"{folder}: {yolo_dataset_path}/{folder}/images\n")
        file.write(f"\nnc: {len(YOLO_CLASSES)}\n")
        file.write(f"\nnames: {list(YOLO_CLASSES.keys())}\n")
    print(f"Done: the YOLOv8 yaml file was successfully created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert parquet files to YOLOv8 format")
    parser.add_argument(
        "--test-parquet",
        type=str,
        default="../../data/SVHN/test.parquet",
        required=False,
        help="Path to test.parquet file",
    )
    parser.add_argument(
        "--train-parquet",
        type=str,
        default="../../data/SVHN/train.parquet",
        required=False,
        help="Path to train.parquet file",
    )
    parser.add_argument(
        "--yolo-dataset-path",
        type=str,
        default="../../data/YoloSVHN",
        required=False,
        help="Path to output YOLOv8 dataset",
    )
    parser.add_argument(
        "--yaml-file-path",
        type=str,
        default="./yolo_svhn.yaml",
        required=False,
        help="Path to output YOLOv8 yaml file",
    )
    args = parser.parse_args()

    # Create dataset structure
    folders = ["train", "test"]
    for folder in folders:
        create_yolov8_dataset_folders(folder, args.yolo_dataset_path)

    # Process the parquet files
    process_parquet_to_yolo(args.train_parquet, "train", args.yolo_dataset_path)
    process_parquet_to_yolo(args.test_parquet, "test", args.yolo_dataset_path)

    # Create YAML config
    create_yolov8_yaml_file(args.yolo_dataset_path, args.yaml_file_path, folders)