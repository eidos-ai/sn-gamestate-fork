import os
import argparse
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm


class SVHNDataset(Dataset):
    def __init__(self, root, split, transform=None):
        total = 0
        capped_total = 0
        self.root = root
        self.transform = transform
        self.data = []

        images_folder = os.path.join(self.root, "images")
        labels_folder = os.path.join(self.root, "labels")

        images_split_folder = os.path.join(images_folder, split)
        labels_split_folder = os.path.join(labels_folder, split)

        for filename in tqdm(
            os.listdir(images_split_folder), desc=f"Loading {split} images"
        ):
            if filename.lower().endswith(".jpg"):
                total += 1
                img_path = os.path.join(images_split_folder, filename)
                label = self._get_label(labels_split_folder, filename)
                # Check if label has maximum two digits
                if len(label) <= 2:
                    capped_total += 1
                    self.data.append({"path": img_path, "label": int(label)})

        print("Total:", total)
        print("Capped:", capped_total)

    def _get_label(self, labels_folder, filename):
        label = ""
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_filepath = os.path.join(labels_folder, txt_filename)
        with open(txt_filepath, "r") as file:
            for line in file:
                digit = line.strip().split()[0]
                label += digit
        return label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]["path"]
        label = self.data[idx]["label"]
        original_image = Image.open(img_path)
        if self.transform:
            image = self.transform(original_image)
        return image, label


def main():
    parser = argparse.ArgumentParser(description="DataLoader for SVHN dataset.")
    parser.add_argument(
        "--dataset_folder", type=str, required=True, help="Path to the dataset folder"
    )
    args = parser.parse_args()

    if not os.path.exists(args.dataset_folder):
        print("Error: Dataset folder does not exist.")
        return

    dataset = SVHNDataset(root=args.dataset_folder, transform=None)

    for idx in tqdm(range(len(dataset)), desc="Iterating over SVHN dataset"):
        item = dataset[idx]


if __name__ == "__main__":
    main()
