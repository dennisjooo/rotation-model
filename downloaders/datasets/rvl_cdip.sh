#!/bin/bash

# Source utilities
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$SCRIPT_DIR/../utils.sh"

# Download RVL-CDIP dataset
download_rvl_cdip() {
    local data_dir="data/rvl-cdip"
    
    if [ -d "$data_dir" ] && [ "$(ls -A $data_dir/*.tif 2>/dev/null)" ]; then
        echo "RVL-CDIP dataset already exists, skipping download..."
        return
    fi
    
    echo "Downloading RVL-CDIP from Kaggle..."
    ensure_dir "$data_dir"
    kaggle datasets download uditamin/rvl-cdip-small -p "$data_dir"
    
    echo "Extracting dataset..."
    unzip -j "$data_dir/rvl-cdip-small.zip" "data/**/*.tif" -d "$data_dir"
    
    echo "Cleaning up..."
    rm "$data_dir/rvl-cdip-small.zip"
}

# If script is run directly, execute the download
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    download_rvl_cdip
fi 