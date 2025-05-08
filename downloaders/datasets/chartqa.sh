#!/bin/bash

# Source utilities
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$SCRIPT_DIR/../utils.sh"

# Download ChartQA dataset
download_chartqa() {
    local data_dir="data/chartqa"
    local temp_dir="$data_dir-temp"
    
    if [ -d "$data_dir" ] && [ "$(ls -A $data_dir/*.jpg 2>/dev/null)" ]; then
        echo "Chart-QA dataset already exists, skipping download..."
        return
    fi
    
    echo "Downloading Chart-QA dataset..."
    ensure_dir "$data_dir"
    
    # Download from HuggingFace
    git clone https://huggingface.co/datasets/HuggingFaceM4/ChartQA "$temp_dir"
    
    # Process the parquet files using the utility script
    python3 "$SCRIPT_DIR/utils.py" "chartqa" "$temp_dir/data" "$data_dir"
    
    # Cleanup
    rm -rf "$temp_dir"
}

# If script is run directly, execute the download
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    download_chartqa
fi 