import os
import argparse

def generate_image_labels(parent_folder, output_file="image_labels.txt"):
    """
    Generate a text file with image paths and their corresponding folder names as labels.
    Skips folders named 'None' or '0'.
    
    Args:
        parent_folder (str): Path to the parent folder containing child folders with images.
        output_file (str): Name of the output text file.
    """
    with open(os.path.join(parent_folder,output_file), 'w') as f:
        for root, dirs, files in os.walk(parent_folder):
            # Skip '.ipynb_checkpoints' directories
            if '.ipynb_checkpoints' in root.split(os.sep):
                continue
            # Get the immediate child folder name
            current_dir = os.path.basename(root)
            
            # Skip if the folder is 'None' or '00'
            if current_dir in ['None', '00']:
                continue
                
            # Process each file in the directory
            for file in files:
                # Check if the file is an image (you can add more extensions if needed)
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    # Get the relative path from the parent folder
                    rel_path = os.path.relpath(os.path.join(root, file), parent_folder)
                    # Replace path separator with '/' for consistency (optional)
                    rel_path = rel_path.replace(os.sep, '/')
                    # Write to the output file
                    f.write(f"{rel_path} {current_dir}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate image labels from folder structure.')
    parser.add_argument('parent_folder', type=str, help='Path to the parent folder containing child folders with images')
    parser.add_argument('--output', type=str, default="image_labels.txt", help='Output text file name (default: image_labels.txt)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.parent_folder):
        print(f"Error: The specified parent folder does not exist: {args.parent_folder}")
    else:
        generate_image_labels(args.parent_folder, args.output)
        print(f"Successfully generated labels in {args.output}")