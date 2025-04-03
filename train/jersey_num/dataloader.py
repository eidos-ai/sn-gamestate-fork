"""
DataLoader to train the Jersey Number Recognition model.
"""

import os
import argparse
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from tqdm import tqdm


class JerseyNumberDataset(Dataset):

    def __init__(self, dataset_folder, transform=None, include_number_not_seen=False):
        self.transform = transform
        self.include_number_not_seen = include_number_not_seen
        self.data = []

        # Iterate through subdirectories
        for label in os.listdir(dataset_folder):
            label_path = os.path.join(dataset_folder, label)

            # Check if it's a directory and not '0' if include_number_not_seen is False
            if os.path.isdir(label_path) and (label != "0" or include_number_not_seen):
                # Iterate through images in the subdirectory
                for img_name in os.listdir(label_path):
                    if img_name.lower().endswith(".jpg"):
                        img_path = os.path.join(label_path, img_name)
                        # Append path and label to the data list
                        self.data.append({"image_path": img_path, "label": int(label)})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get item at position idx
        item = self.data[idx]

        # Get path and label from the item
        img_path = item["image_path"]
        label = item["label"]

        # Open image using PIL
        image = Image.open(img_path)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    parser = argparse.ArgumentParser(description="DataLoader for JerseyNumberDataset.")
    parser.add_argument(
        "--dataset_folder", type=str, required=True, help="Path to the dataset folder"
    )
    parser.add_argument(
        "--include_none", action="store_true", help="Include number 0 if set"
    )
    args = parser.parse_args()

    # Validate if dataset folder exists
    if not os.path.exists(args.dataset_folder):
        print("Error: Dataset folder does not exist.")
        return

    dataset = JerseyNumberDataset(
        dataset_folder=args.dataset_folder, include_number_not_seen=args.include_none
    )

    # Iterate over the dataset using tqdm
    for idx in tqdm(range(len(dataset))):
        item = dataset[idx]


if __name__ == "__main__":
    main()
