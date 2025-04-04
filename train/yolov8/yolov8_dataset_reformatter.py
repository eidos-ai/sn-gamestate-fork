import os
import json
import shutil
import argparse
from tqdm import tqdm  # type: ignore

YOLO_CLASSES = {"person": 0}
FRAME_SIZE = {
    "width": 1920,
    "height": 1080,
}


def create_yolov8_dataset_folders(folder: str, yolo_dataset_path: str) -> None:
    """Create the YOLOv8 dataset folders"""
    os.makedirs(f"{yolo_dataset_path}/{folder}/images", exist_ok=True)
    os.makedirs(f"{yolo_dataset_path}/{folder}/labels", exist_ok=True)
    print(f"Done: the YOLOv8 dataset {folder} folders were successfully created.")


def format_images(folder: str, yolo_dataset_path: str, raw_dataset_path: str) -> None:
    if folder == "val":
        folder = "valid"
    folder_path = os.path.join(raw_dataset_path, folder)
    for root, dirs, files in tqdm(os.walk(folder_path)):
        for document in files:
            if document.endswith(".jpg"):
                image_path = os.path.join(root, document)
                new_image_name = ""
                index = image_path.find("SNGS-")
                if index != -1:
                    new_image_name = (
                        image_path[index : index + 8] + "-" + image_path[-10:-4]
                    )
                else:
                    exit()
                if folder == "valid":
                    folder = "val"
                new_image_path = (
                    f"{yolo_dataset_path}/{folder}/images/{new_image_name}.jpg"
                )
                shutil.copy2(os.path.join(root, document), new_image_path)
    print(
        f"Done: the images were successfully exported into {folder} folder in the YOLO dataset."
    )


def create_annotation_txt_file(
    folder: str, yolo_dataset_path: str, raw_dataset_path: str
) -> None:
    """from Labels-GameState.json, get the annotations of the corresponding image and
    write them to a txt file in YOLOv8 format"""
    if folder == "val":
        folder = "valid"
    folder_path = os.path.join(raw_dataset_path, folder)
    for root, dirs, files in tqdm(os.walk(folder_path)):
        for document in files:
            if document.endswith("Labels-GameState.json"):
                with open(os.path.join(root, document)) as json_file:
                    data = json.load(json_file)
                image_ids = set()

                for annotation in data["annotations"]:
                    image_ids.add(annotation["image_id"])
                for image_id in image_ids:
                    formatted_annotations = []
                    for annotation in data["annotations"]:
                        if annotation["image_id"] == image_id and (
                            annotation["category_id"] == 1
                            or annotation["category_id"] == 2
                            or annotation["category_id"] == 3
                        ):
                            formatted_annotation = (
                                f"{YOLO_CLASSES['person']} "
                                + str(
                                    (float(annotation["bbox_image"]["x_center"]))
                                    / FRAME_SIZE["width"]
                                )
                                + " "
                                + str(
                                    (float(annotation["bbox_image"]["y_center"]))
                                    / FRAME_SIZE["height"]
                                )
                                + " "
                                + str(
                                    (float(annotation["bbox_image"]["w"]))
                                    / FRAME_SIZE["width"]
                                )
                                + " "
                                + str(
                                    (float(annotation["bbox_image"]["h"]))
                                    / FRAME_SIZE["height"]
                                )
                            )
                            formatted_annotations.append(formatted_annotation)
                    json_file_path = os.path.join(root, document)
                    new_txt_name = ""
                    index = json_file_path.find("SNGS-")
                    if index != -1:
                        new_txt_name = (
                            json_file_path[index : index + 8]
                            + "-"
                            + image_id[-6:]
                            + ".txt"
                        )
                    else:
                        exit()
                    if folder == "valid":
                        folder = "val"
                    new_txt_path = f"{yolo_dataset_path}/{folder}/labels/{new_txt_name}"
                    with open(new_txt_path, "w") as file:
                        for ann in formatted_annotations:
                            file.write(ann + "\n")
    print(
        f"Done: the labels txt files were successfully exported into {folder} folder in the YOLO dataset."
    )


def create_yolov8_yaml_file(
    yolo_dataset_path: str, yaml_file_path: str, folders: list[str]
) -> None:
    """Create the YOLOv8 yaml file"""
    with open(yaml_file_path, "w") as file:
        for folder in folders:
            file.write(
                f"{folder}: ./GameStateChallenge/{yolo_dataset_path}/{folder}/images\n"
            )
        file.write(f"\nnc: {len(YOLO_CLASSES)}\n")
        file.write(f"\nnames: {list(YOLO_CLASSES.keys())}\n")
    print(f"Done: the YOLOv8 yaml file was successfully created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-rdp",
        "--raw-dataset-path",
        type=str,
        required=True,
        help="Path of the raw dataset.",
    )
    parser.add_argument(
        "-ydp",
        "--yolo-dataset-path",
        type=str,
        required=True,
        help="Path of the new YOLOv8 dataset.",
    )
    parser.add_argument(
        "-yaml",
        "--yaml-file-path",
        type=str,
        required=True,
        help="Path of the YOLOv8 yaml file.",
    )
    args = parser.parse_args()
    raw_dataset_path = args.raw_dataset_path
    yolo_dataset_path = args.yolo_dataset_path
    yaml_file_path = args.yaml_file_path

    folders = ["train", "test", "val"]
    create_yolov8_yaml_file(yolo_dataset_path, yaml_file_path, folders)
    for folder in folders:
        create_yolov8_dataset_folders(folder, yolo_dataset_path)
        format_images(folder, yolo_dataset_path, raw_dataset_path)
        create_annotation_txt_file(folder, yolo_dataset_path, raw_dataset_path)
