# Create target directory
mkdir -p merged_dataset

# Copy all files from all source directories, merging contents
for src_dir in train_0.85 valid_0.85; do
    # Find all numbered directories
    find "$src_dir" -type d -name "[0-9]*" | while read -r dirpath; do
        # Get just the number (last part of path)
        number=$(basename "$dirpath")
        # Create target directory if needed
        mkdir -p "merged_dataset/$number"
        # Copy all files
        cp -n "$dirpath"/* "merged_dataset/$number/" 2>/dev/null
    done
done