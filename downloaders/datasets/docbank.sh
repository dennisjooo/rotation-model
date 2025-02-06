#!/bin/bash

# Source utilities
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$SCRIPT_DIR/../utils.sh"

# Download DocBank dataset
download_docbank() {
    local data_dir="data/docbank"
    
    if [ -d "$data_dir" ] && [ "$(ls -A $data_dir/*.pdf 2>/dev/null)" ]; then
        echo "DocBank dataset already exists, skipping download..."
        return
    fi
    
    echo "Downloading DocBank dataset..."
    ensure_dir "$data_dir"
    
    # Download from official source
    wget https://github.com/doc-analysis/DocBank/raw/master/DocBank_samples/DocBank_samples.zip -P "$data_dir/"
    unzip "$data_dir/DocBank_samples.zip" -d "$data_dir/"
    rm "$data_dir/DocBank_samples.zip"
}

# If script is run directly, execute the download
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    download_docbank
fi 