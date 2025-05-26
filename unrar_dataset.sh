#!/bin/bash
# filepath: /workspace/Code/unrar_dataset.sh

# Check if unrar is installed
if ! command -v unrar &> /dev/null; then
    echo "Error: unrar is not installed. Please install it first."
    echo "You can install it using: sudo apt-get install unrar"
    exit 1
fi

# Check if directory argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory_path>"
    echo "Example: $0 /path/to/rarfiles"
    exit 1
fi

SOURCE_DIR="$1"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Directory '$SOURCE_DIR' does not exist."
    exit 1
fi

echo "Searching for RAR files in '$SOURCE_DIR'..."
COUNT=0
FAILED=0

# Find all .rar files and extract them to their own directory
find "$SOURCE_DIR" -type f \( -name "*.rar" -o -name "*.RAR" \) | while read -r rarfile; do
    # Get the directory where the RAR file is located
    extract_dir="$(dirname "$rarfile")"
    
    echo "Extracting: $(basename "$rarfile")"
    echo "Location: $extract_dir"
    
    # Extract to the same directory where the RAR file is located
    if unrar x -o+ "$rarfile" "$extract_dir"; then
        echo "✓ Extracted: $(basename "$rarfile")"
        ((COUNT++))
    else
        echo "✗ Failed to extract: $(basename "$rarfile")"
        ((FAILED++))
    fi
    echo "---------------------------------"
done

echo "Extraction complete!"
echo "$COUNT files extracted successfully."
[ $FAILED -gt 0 ] && echo "$FAILED files failed to extract."