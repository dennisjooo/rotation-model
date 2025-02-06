#!/bin/bash

# Source utilities
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$SCRIPT_DIR/../utils.sh"

# Download PubLayNet dataset
download_publaynet() {
    local data_dir="data/publaynet"
    local temp_dir="$data_dir-temp"
    
    if [ -d "$data_dir" ] && [ "$(ls -A $data_dir/*.png 2>/dev/null)" ]; then
        echo "PubLayNet dataset already exists and is extracted, skipping..."
        return
    fi
    
    echo "Downloading PubLayNet from HuggingFace..."
    ensure_dir "$data_dir"
    git clone https://huggingface.co/datasets/lhoestq/small-publaynet-wds "$temp_dir"
    mv "$temp_dir/publaynet-train-"*.tar "$data_dir/" || {
        echo "Error: No publaynet tar files found in the downloaded repository"
        rm -rf "$temp_dir"
        exit 1
    }
    rm -rf "$temp_dir"
    
    echo "Extracting PubLayNet tar files..."
    for tarfile in "$data_dir"/*.tar; do
        tar xf "$tarfile" -C "$data_dir/"
        rm "$tarfile"
    done

    echo "Cleaning up non-image files..."
    find "$data_dir" -type f ! -name "*.png" -delete
}

# If script is run directly, execute the download
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    download_publaynet
fi 