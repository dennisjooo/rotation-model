#!/bin/bash

# Source utilities
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$SCRIPT_DIR/../utils.sh"

# Download TableBench dataset
download_tablebench() {
    local data_dir="data/tablebench"
    
    if [ -d "$data_dir" ] && [ "$(ls -A $data_dir/*.png 2>/dev/null)" ]; then
        echo "TableBench dataset already exists, skipping download..."
        return
    fi
    
    echo "Downloading TableBench dataset..."
    ensure_dir "$data_dir"
    
    # Download from HuggingFace
    wget "https://huggingface.co/datasets/reducto/rd-tablebench/resolve/main/rd-tablebench.zip?download=true" -O "$data_dir/rd-tablebench.zip"
    
    echo "Extracting dataset..."
    unzip "$data_dir/rd-tablebench.zip" -d "$data_dir/"
    
    echo "Moving all PNG images to root directory..."
    find "$data_dir/rd-tablebench" -type f -name "*.png" -exec mv {} "$data_dir/" \;
    
    echo "Cleaning up..."
    rm "$data_dir/rd-tablebench.zip"
    rm -rf "$data_dir/rd-tablebench"
}

# If script is run directly, execute the download
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    download_tablebench
fi 