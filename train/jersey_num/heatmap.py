import os
import numpy as np
import matplotlib.pyplot as plt
import argparse


def generate_heatmap(directory: str, name: str) -> None:
    def count_crops(directory):
        crop_counts = np.zeros((10, 10), dtype=int)
        total_crops = 0
        for i in range(0, 100):
            folder_name = str(i)
            folder_path = os.path.join(directory, folder_name)
            # print(folder_path, os.path.exists(folder_path))
            if os.path.exists(folder_path):
                num_crops = len(
                    [
                        name
                        for name in os.listdir(folder_path)
                        if os.path.isfile(os.path.join(folder_path, name))
                    ]
                )
                tens_place = i // 10
                units_place = i % 10
                crop_counts[tens_place, units_place] = num_crops
                total_crops += num_crops
        return crop_counts, total_crops

    # Count the number of crops in each folder and total number of crops
    data, total_crops = count_crops(directory)

    # Create a heatmap with numbers on each cell and separated borders
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Number of Crops")
    plt.title("Number of Crops in Each Folder\nTotal Crops: {}".format(total_crops))
    plt.xlabel("Units")
    plt.ylabel("Tens")
    plt.xticks(np.arange(10) - 0.5, range(10))
    plt.yticks(np.arange(10) - 0.5, range(10))
    plt.grid(True, linewidth=2, color="black")

    # Add text annotations
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(data[i, j]), ha="center", va="center", color="white")

    # Save the plot as an image file
    plt.savefig(name)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate the jersey heatmap for a dataset"
    )
    parser.add_argument("dataset", type=str, help="Dataset path")
    parser.add_argument(
        "--name", type=str, help="Output file name (stats.png)", default="stats.png"
    )
    args = parser.parse_args()

    # Check if input dataset path exists
    if not os.path.exists(args.dataset):
        print(f"Error: Input dataset path '{args.dataset}' does not exist.")
        return

    generate_heatmap(args.dataset, args.name)


if __name__ == "__main__":
    main()
