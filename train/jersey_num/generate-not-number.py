import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse


def count_annotations(input_dataset_path: str) -> int:
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

            images = {img["image_id"]: img["file_name"] for img in data["images"]}
            annotations = [
                ann
                for ann in data["annotations"]
                if "attributes" in ann
                and (
                    ann["attributes"]["role"] == "player"
                    or ann["attributes"]["role"] == "goalkeeper"
                )
            ]
            annotations.sort(key=lambda ann: int(ann["image_id"]), reverse=False)

            last_processed_frame = {}

            for annotation in annotations:
                pbar.update(1)
                image_id = annotation["image_id"]
                track_id = annotation["track_id"]
                # print(track_id)

                if track_id is None:
                    continue

                if (
                    track_id in last_processed_frame
                    and (int(image_id) - last_processed_frame[track_id])
                    < cooldown_frames
                ):
                    # print('skipping: image - track', image_id, track_id)
                    continue  # Skip processing this annotation due to cooldown

                last_processed_frame[track_id] = int(image_id)

                image_file_name = images.get(image_id, None)

                if image_file_name is None:
                    continue

                image_path = os.path.join(folder_path, "img1", image_file_name)
                image = Image.open(image_path)

                bbox = annotation["bbox_image"]
                cropped_image = image.crop(
                    (bbox["x"], bbox["y"], bbox["x"] + bbox["w"], bbox["y"] + bbox["h"])
                )
                #img_array = np.array(cropped_image)
                #print(img_array.shape)
                #raise Exception
                jersey_number = str(annotation["attributes"].get("jersey", "0"))
                output_folder = os.path.join(output_dataset_path, jersey_number)
                os.makedirs(output_folder, exist_ok=True)
                output_file_path = os.path.join(
                    output_folder, f"{annotation['id']}.jpg"
                )
                if not os.path.exists(output_file_path):
                    output_folder = os.path.join(output_dataset_path, "00")
                    os.makedirs(output_folder, exist_ok=True)
                    output_file_path = os.path.join(
                    output_folder, f"{annotation['id']}.jpg"
                    )
                    cropped_image.save(output_file_path)

def main():
    parser = argparse.ArgumentParser(description="Generate jersey number dataset")
    parser.add_argument("input_path", type=str, help="Input dataset path")
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output dataset path. (./dataset)",
        default="dataset",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Threshold for SVHN detection. (0.55)",
        default=0.55,
    )
    parser.add_argument(
        "--cooldown_frames",
        type=int,
        help="Cooldown frames for player detection. (5)",
        default=5,
    )
    args = parser.parse_args()

    # Check if input dataset path exists
    if not os.path.exists(args.input_path):
        print(f"Error: Input dataset path '{args.input_path}' does not exist.")
        return

    # Process the dataset
    process_dataset(
        args.input_path,
        args.output_path,
        args.threshold,
        args.cooldown_frames,
    )


if __name__ == "__main__":
    main()
