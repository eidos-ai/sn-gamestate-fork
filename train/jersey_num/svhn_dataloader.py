from pathlib import Path
import argparse
from typing import Optional, Dict, List, Tuple, Any
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import torch


class SVHNDataset(Dataset):
    def __init__(self, root: Path, split: str, transform: Optional[Any] = None) -> None:
        """
        SVHN Dataset loader with support for digit length capping.

        Args:
            root: Path to the root directory containing 'images' and 'labels' folders
            split: Dataset split ('train', 'test', or 'valid')
            transform: Optional torchvision transforms to apply
        """
        self.root = root
        self.transform = transform
        self.data: List[Dict[str, Any]] = []
        
        # Validate paths
        images_folder = self.root / "images"
        labels_folder = self.root / "labels"
        if not images_folder.exists() or not labels_folder.exists():
            raise FileNotFoundError("Dataset folder structure invalid - must contain 'images' and 'labels' subfolders")

        images_split_folder = images_folder / split
        labels_split_folder = labels_folder / split

        total = 0
        capped_total = 0

        for img_file in tqdm(
            images_split_folder.glob("*.jpg"), 
            desc=f"Loading {split} images"
        ):
            total += 1
            label = self._get_label(labels_split_folder, img_file.stem)
            
            # Only keep samples with 1-2 digit labels
            if 1 <= len(label) <= 2:
                capped_total += 1
                self.data.append({
                    "path": img_file,
                    "label": int(label)
                })

        print(f"Total images: {total}")
        print(f"Images with 1-2 digit labels: {capped_total}")

    def _get_label(self, labels_folder: Path, filename_stem: str) -> str:
        """
        Extract label from corresponding text file.
        
        Args:
            labels_folder: Path to labels directory
            filename_stem: Filename without extension
            
        Returns:
            Concatenated digits as string
        """
        txt_file = labels_folder / f"{filename_stem}.txt"
        label = []
        
        try:
            with txt_file.open('r') as f:
                for line in f:
                    digit = line.strip().split()[0]
                    label.append(digit)
        except FileNotFoundError:
            print(f"Warning: Missing label file {txt_file}")
            return ""
            
        return "".join(label)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get dataset item by index.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple of (image_tensor, label)
        """
        item = self.data[idx]
        img = Image.open(item["path"])
        
        if self.transform:
            img = self.transform(img)
            
        return img, item["label"]


def main() -> None:
    """
    Command-line interface for SVHN dataset loader.
    
    Usage:
        python svhn_loader.py --dataset_folder /path/to/svhn
    """
    parser = argparse.ArgumentParser(description="DataLoader for SVHN dataset.")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        required=True,
        help="Path to the dataset folder containing 'images' and 'labels' subfolders"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_folder)
    if not dataset_path.exists():
        print(f"Error: Dataset folder does not exist at {dataset_path}")
        return

    try:
        dataset = SVHNDataset(root=dataset_path, split="train", transform=None)
        
        for idx in tqdm(range(len(dataset)), desc="Iterating over SVHN dataset"):
            _ = dataset[idx]  # Process each item
            
    except Exception as e:
        print(f"Error loading dataset: {e}")


if __name__ == "__main__":
    main()