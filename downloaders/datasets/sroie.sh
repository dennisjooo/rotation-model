#!/bin/bash

# Source utilities
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$SCRIPT_DIR/../utils.sh"

# Download SROIE dataset
download_sroie() {
    local data_dir="data/sroie"
    local temp_dir="$data_dir/temp"
    
    if [ -d "$data_dir" ] && [ "$(ls -A $data_dir/*.jpg 2>/dev/null)" ]; then
        echo "SROIE dataset already exists, skipping download..."
        return
    fi
    
    echo "Downloading SROIE from Kaggle..."
    ensure_dir "$temp_dir"
    kaggle datasets download urbikn/sroie-datasetv2 -p "$temp_dir"
    
    echo "Extracting dataset..."
    unzip "$temp_dir/sroie-datasetv2.zip" -d "$temp_dir"
    
    echo "Moving all images to root directory..."
    find "$temp_dir/SROIE2019" -type f -path "*/img/*.jpg" -exec mv {} "$data_dir/" \;
    
    echo "Cleaning up..."
    rm -rf "$temp_dir"
}

# If script is run directly, execute the download
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    download_sroie
fi 