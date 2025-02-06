#!/bin/bash

# Source utilities
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$SCRIPT_DIR/../utils.sh"

# Download CORD dataset
download_cord() {
    local data_dir="data/cord"
    local temp_dir="$data_dir-temp"
    
    if [ -d "$data_dir" ] && [ "$(ls -A $data_dir/*.jpg 2>/dev/null)" ]; then
        echo "CORD dataset already exists, skipping download..."
        return
    fi
    
    echo "Downloading CORD dataset..."
    ensure_dir "$data_dir"
    
    # Download from HuggingFace
    git clone https://huggingface.co/datasets/cord-v2 "$temp_dir"
    mv "$temp_dir/train/"* "$data_dir/"
    rm -rf "$temp_dir"
}

# If script is run directly, execute the download
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    download_cord
fi 