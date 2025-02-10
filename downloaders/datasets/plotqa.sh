#!/bin/bash

# Source utilities
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$SCRIPT_DIR/../utils.sh"

# Download PlotQA dataset
download_plotqa() {
    local data_dir="data/plotqa"
    
    if [ -d "$data_dir" ] && [ "$(ls -A $data_dir/*.png 2>/dev/null)" ]; then
        echo "PlotQA dataset already exists, skipping download..."
        return
    fi
    
    echo "Downloading PlotQA dataset..."
    ensure_dir "$data_dir"
    
    # Download from Google Drive using gdown
    echo "Downloading PlotQA dataset from Google Drive..."
    gdown https://drive.google.com/uc?id=1i74NRCEb-x44xqzAovuglex5d583qeiF -O "$data_dir/png.tar.gz"
    
    echo "Extracting dataset..."
    tar -xzf "$data_dir/png.tar.gz" -C "$data_dir/"
    rm "$data_dir/png.tar.gz"
    
    # Move all PNG images to root directory if they're in subdirectories
    find "$data_dir" -type f -name "*.png" -exec mv {} "$data_dir/" \;
    
    # Clean up any empty subdirectories
    find "$data_dir" -type d -empty -delete
}

# If script is run directly, execute the download
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    download_plotqa
fi 