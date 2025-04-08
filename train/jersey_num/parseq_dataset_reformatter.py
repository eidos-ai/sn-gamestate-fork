from pathlib import Path
import argparse
from typing import List, Optional

def generate_image_labels(parent_folder: Path, output_file: str = "image_labels.txt") -> None:
    """
    Generate a text file with image paths and their corresponding folder names as labels.
    Skips folders named 'None' or '00' and Jupyter notebook checkpoints.
    
    Args:
        parent_folder: Path to the parent folder containing child folders with images.
        output_file: Name of the output text file (will be created in parent_folder).
    
    Returns:
        None (writes output to file)
    """
    # Supported image extensions
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    
    with open(parent_folder / output_file, 'w') as f:
        for child_dir in parent_folder.iterdir():
            if not child_dir.is_dir():
                continue
                
            # Skip special directories
            if child_dir.name in {'None', '00'} or '.ipynb_checkpoints' in child_dir.parts:
                continue
                
            # Process each image file in the directory
            for image_file in child_dir.glob('*'):
                if image_file.suffix.lower() in IMAGE_EXTENSIONS:
                    # Get relative path using forward slashes
                    rel_path = image_file.relative_to(parent_folder).as_posix()
                    f.write(f"{rel_path} {child_dir.name}\n")

def main() -> None:
    """
    Command-line interface for generating image labels from folder structure.
    
    Usage:
        python generate_labels.py /path/to/parent_folder [--output labels.txt]
    """
    parser = argparse.ArgumentParser(
        description='Generate image labels from folder structure.'
    )
    parser.add_argument(
        'parent_folder', 
        type=str, 
        help='Path to the parent folder containing child folders with images'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default="image_labels.txt",
        help='Output text file name (default: image_labels.txt)'
    )
    
    args = parser.parse_args()
    parent_path = Path(args.parent_folder)
    
    if not parent_path.is_dir():
        print(f"Error: The specified parent folder does not exist: {parent_path}")
    else:
        generate_image_labels(parent_path, args.output)
        output_path = parent_path / args.output
        print(f"Successfully generated labels in {output_path}")

if __name__ == "__main__":
    main()