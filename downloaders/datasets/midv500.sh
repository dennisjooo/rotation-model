#!/bin/bash

# Source utilities
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$SCRIPT_DIR/../utils.sh"

# Download MIDV-500 dataset
download_midv500() {
    local data_dir="data/midv500"
    local temp_dir="$data_dir/temp"
    
    if [ -d "$data_dir" ] && [ "$(ls -A $data_dir/*.tif 2>/dev/null)" ]; then
        echo "MIDV-500 dataset already exists, skipping download..."
        return
    fi
    
    echo "Downloading MIDV-500 using Python script..."
    ensure_dir "$temp_dir"
    python "$SCRIPT_DIR/../../downloaders/download_midv.py" --dir "$temp_dir" --yes
    
    echo "Moving all TIFF images to root directory..."
    find "$temp_dir" -type f -name "*.tif" -exec mv {} "$data_dir/" \;
    
    echo "Cleaning up..."
    rm -rf "$temp_dir"
}

# If script is run directly, execute the download
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    download_midv500
fi 