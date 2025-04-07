import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse
from typing import Dict, List, Optional


def count_annotations(input_dataset_path: str) -> int:
    """
    Count the total number of player and goalkeeper annotations in the dataset.

    Args:
        input_dataset_path: Path to the root directory containing SNGS folders with annotation files.

    Returns:
        Total count of player and goalkeeper annotations found in all JSON files.
    """
    total_annotations = 0

    # Get list of folders containing 'SNGS' in their name
    folders = [
        folder_name
        for folder_name in os.listdir(input_dataset_path)
        if "SNGS" in folder_name
    ]

    for folder_name in folders:
        folder_path = os.path.join(input_dataset_path, folder_name)
        json_path = os.path.join(folder_path, "Labels-GameState.json")

        # Check if JSON file exists
        if not os.path.isfile(json_path):
            print(f"Error: JSON file '{json_path}' not found.")
            continue

        try:
            with open(json_path, "r") as json_file:
                data = json.load(json_file)
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
            and (
                ann["attributes"]["role"] == "player"
                or ann["attributes"]["role"] == "goalkeeper"
            )
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

    # Get list of folders containing 'SNGS' in their name
    folders = [
        folder_name
        for folder_name in os.listdir(input_dataset_path)
        if "SNGS" in folder_name
    ]

    with tqdm(total=total_annotations, desc="Processing") as pbar:
        for folder_name in folders:
            folder_path = os.path.join(input_dataset_path, folder_name)
            json_path = os.path.join(folder_path, "Labels-GameState.json")

            # Check if JSON file exists
            if not os.path.isfile(json_path):
                print(f"Error: JSON file '{json_path}' not found.")
                continue

            try:
                with open(json_path, "r") as json_file:
                    data = json.load(json_file)
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

                image_path: str = os.path.join(folder_path, "img1", image_file_name)
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
                output_folder: str = os.path.join(output_dataset_path, jersey_number)
                os.makedirs(output_folder, exist_ok=True)
                output_file_path: str = os.path.join(
                    output_folder, f"{annotation['id']}.jpg"
                )
                
                if not os.path.exists(output_file_path):
                    try:
                        cropped_image.save(output_file_path)
                    except Exception as e:
                        print(f"Error saving image {output_file_path}: {e}")
                        # Fallback to saving in "00" folder if there's an error
                        output_folder = os.path.join(output_dataset_path, "00")
                        os.makedirs(output_folder, exist_ok=True)
                        output_file_path = os.path.join(
                            output_folder, f"{annotation['id']}.jpg"
                        )
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

    # Check if input dataset path exists
    if not os.path.exists(args.input_path):
        print(f"Error: Input dataset path '{args.input_path}' does not exist.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Process the dataset
    process_dataset(
        args.input_path,
        args.output_path,
        args.threshold,
        args.cooldown_frames,
    )


if __name__ == "__main__":
    main()