from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Tuple


def count_crops(directory: Path) -> Tuple[np.ndarray, int]:
    """
    Count the number of image crops in each jersey number folder.

    Args:
        directory: Path to the dataset directory containing numbered folders (00-99)

    Returns:
        Tuple containing:
        - 10x10 numpy array with crop counts (tens place x units place)
        - Total number of crops across all folders
    """
    crop_counts = np.zeros((10, 10), dtype=int)
    total_crops = 0
    
    for i in range(100):
        folder = directory / f"{i:02d}"  # Format with leading zero
        if folder.exists() and folder.is_dir():
            num_crops = sum(1 for _ in folder.glob("*") if _.is_file())
            tens, units = divmod(i, 10)
            crop_counts[tens, units] = num_crops
            total_crops += num_crops
            
    return crop_counts, total_crops


def generate_heatmap(directory: str, name: str) -> None:
    """
    Generate and save a heatmap visualization of jersey number distribution.

    Args:
        directory: Path to the dataset directory
        name: Output filename for the heatmap image
    """
    dir_path = Path(directory)
    data, total_crops = count_crops(dir_path)

    # Create and configure the heatmap plot
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Number of Crops")
    plt.title(f"Jersey Number Distribution\nTotal Crops: {total_crops}")
    plt.xlabel("Units Digit")
    plt.ylabel("Tens Digit")
    
    # Set ticks and grid
    plt.xticks(np.arange(10) - 0.5, range(10))
    plt.yticks(np.arange(10) - 0.5, range(10))
    plt.grid(True, linewidth=2, color="black")

    # Add count annotations to each cell
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(data[i, j]), 
                    ha="center", va="center", 
                    color="white" if data[i, j] > data.max()/2 else "black")

    # Save the plot
    output_path = dir_path / name
    plt.savefig(output_path)
    plt.close()
    print(f"Heatmap saved to: {output_path}")


def main() -> None:
    """
    Command-line interface for generating jersey number heatmaps.
    
    Usage:
        python jersey_heatmap.py dataset_path [--name output_filename.png]
    """
    parser = argparse.ArgumentParser(
        description="Generate heatmap visualization of jersey number distribution"
    )
    parser.add_argument(
        "dataset", 
        type=str, 
        help="Path to dataset directory containing numbered folders (00-99)"
    )
    parser.add_argument(
        "--name", 
        type=str, 
        help="Output filename (default: stats.png)", 
        default="stats.png"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return

    generate_heatmap(args.dataset, args.name)


if __name__ == "__main__":
    main()