import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse


def count_annotations(input_dataset_path: str) -> int:
    """
    Count the total number of player and goalkeeper annotations in the dataset.

    Args:
        input_dataset_path: Path to the root directory containing SNGS folders with annotation files.

    Returns:
        Total count of player and goalkeeper annotations found in all JSON files.
    """
    total_annotations = 0
    input_path = Path(input_dataset_path)

    # Get list of folders containing 'SNGS' in their name
    folders = [
        folder for folder in input_path.iterdir() 
        if folder.is_dir() and "SNGS" in folder.name
    ]

    for folder in folders:
        json_path = folder / "Labels-GameState.json"

        # Check if JSON file exists
        if not json_path.is_file():
            print(f"Error: JSON file '{json_path}' not found.")
            continue

        try:
            data = json.loads(json_path.read_text())
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file '{json_path}': {e}")
            continue

        # Check if JSON file contains required keys
        if "images" not in data or "annotations" not in data:
            print(
                f"Error: JSON file '{json_path}' does not contain 'images' or 'annotations' key."
            )
            continue

        annotations = [
            ann
            for ann in data["annotations"]
            if "attributes" in ann
            and ann["attributes"]["role"] in ("player", "goalkeeper")
        ]
        total_annotations += len(annotations)

    return total_annotations


def process_dataset(
    input_dataset_path: str,
    output_dataset_path: str,
    threshold: float,
    cooldown_frames: int,
) -> None:
    """
    Process the soccer dataset to extract player jersey number images.

    The function:
    1. Processes all SNGS folders in the input path
    2. For each player/goalkeeper annotation:
       - Skips if track_id is None
       - Skips if within cooldown_frames of last processing for this track_id
       - Crops the player image using the bounding box
       - Saves the image in a folder named after the jersey number
       - If jersey number is missing or 0, saves in "00" folder
       - If the target file already exists, skips saving

    Note: The threshold parameter is currently unused in the function.

    Args:
        input_dataset_path: Path to directory containing SNGS folders with JSON annotations
        output_dataset_path: Path where cropped images will be saved (organized by number)
        threshold: Unused parameter (kept for interface compatibility)
        cooldown_frames: Minimum frames between processing same player (redundancy control)
    """
    total_annotations = count_annotations(input_dataset_path)
    input_path = Path(input_dataset_path)
    output_path = Path(output_dataset_path)

    # Get list of folders containing 'SNGS' in their name
    folders = [
        folder for folder in input_path.iterdir() 
        if folder.is_dir() and "SNGS" in folder.name
    ]

    with tqdm(total=total_annotations, desc="Processing") as pbar:
        for folder in folders:
            json_path = folder / "Labels-GameState.json"

            # Check if JSON file exists
            if not json_path.is_file():
                print(f"Error: JSON file '{json_path}' not found.")
                continue

            try:
                data = json.loads(json_path.read_text())
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON file '{json_path}': {e}")
                continue

            # Check if JSON file contains required keys
            if "images" not in data or "annotations" not in data:
                print(
                    f"Error: JSON file '{json_path}' does not contain 'images' or 'annotations' key."
                )
                continue

            images: Dict[str, str] = {img["image_id"]: img["file_name"] for img in data["images"]}
            annotations: List[Dict] = [
                ann
                for ann in data["annotations"]
                if "attributes" in ann
                and (
                    ann["attributes"]["role"] == "player"
                    or ann["attributes"]["role"] == "goalkeeper"
                )
            ]
            annotations.sort(key=lambda ann: int(ann["image_id"]), reverse=False)

            last_processed_frame: Dict[str, int] = {}

            for annotation in annotations:
                pbar.update(1)
                image_id: str = annotation["image_id"]
                track_id: Optional[str] = annotation.get("track_id")

                if track_id is None:
                    continue

                if (
                    track_id in last_processed_frame
                    and (int(image_id) - last_processed_frame[track_id])
                    < cooldown_frames
                ):
                    continue  # Skip processing this annotation due to cooldown

                last_processed_frame[track_id] = int(image_id)

                image_file_name: Optional[str] = images.get(image_id)

                if image_file_name is None:
                    continue

                image_path = folder / "img1" / image_file_name
                try:
                    image: Image.Image = Image.open(image_path)
                except Exception as e:
                    print(f"Error opening image {image_path}: {e}")
                    continue

                bbox: Dict[str, float] = annotation["bbox_image"]
                try:
                    cropped_image: Image.Image = image.crop(
                        (bbox["x"], bbox["y"], bbox["x"] + bbox["w"], bbox["y"] + bbox["h"])
                    )
                except Exception as e:
                    print(f"Error cropping image {image_path}: {e}")
                    continue

                jersey_number: str = str(annotation["attributes"].get("jersey", "0"))
                output_folder = output_path / jersey_number
                output_folder.mkdir(parents=True, exist_ok=True)
                output_file_path = output_folder / f"{annotation['id']}.jpg"
                
                if not output_file_path.exists():
                    try:
                        cropped_image.save(output_file_path)
                    except Exception as e:
                        print(f"Error saving image {output_file_path}: {e}")
                        # Fallback to saving in "00" folder if there's an error
                        output_folder = output_path / "00"
                        output_folder.mkdir(parents=True, exist_ok=True)
                        output_file_path = output_folder / f"{annotation['id']}.jpg"
                        try:
                            cropped_image.save(output_file_path)
                        except Exception as e:
                            print(f"Error saving fallback image {output_file_path}: {e}")


def main() -> None:
    """
    Main function to parse arguments and process the soccer jersey number dataset.
    """
    parser = argparse.ArgumentParser(description="Generate jersey number dataset from soccer video frames")
    parser.add_argument(
        "input_path", 
        type=str, 
        help="Path to the input dataset containing SNGS folders with JSON annotations"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path where the processed jersey number images will be saved (default: ./dataset)",
        default="dataset",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Confidence threshold for detection (currently unused, default: 0.55)",
        default=0.55,
    )
    parser.add_argument(
        "--cooldown_frames",
        type=int,
        help="Number of frames to skip after processing a player to avoid duplicates (default: 5)",
        default=5,
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    # Check if input dataset path exists
    if not input_path.exists():
        print(f"Error: Input dataset path '{input_path}' does not exist.")
        return

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Process the dataset
    process_dataset(
        str(input_path),  # Convert to string for backward compatibility
        str(output_path), # Convert to string for backward compatibility
        args.threshold,
        args.threshold,
    )


if __name__ == "__main__":
    main()