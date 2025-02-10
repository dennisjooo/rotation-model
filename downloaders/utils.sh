#!/bin/bash

# Get the directory where the script is located
get_script_dir() {
    echo "$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
}

check_dependencies() {
    # Check if kaggle is installed, install if missing
    if ! command -v kaggle &> /dev/null; then
        echo "Kaggle CLI not found. Installing..."
        pip install kaggle
    fi

    # Check and install Git LFS
    if ! command -v git-lfs &> /dev/null; then
        echo "git-lfs not found. Installing..."
        sudo apt-get install git-lfs
    fi

    # Check and install unzip
    if ! command -v unzip &> /dev/null; then
        echo "unzip not found. Installing..."
        sudo apt-get install unzip
    fi
    
    # Check and install gdown for Google Drive downloads
    if ! command -v gdown &> /dev/null; then
        echo "gdown not found. Installing..."
        pip install gdown
    fi
}

setup_kaggle_auth() {
    echo "Setting up Kaggle credentials..."
    mkdir -p ~/.kaggle
    if [ -f kaggle.json ]; then
        cp kaggle.json ~/.kaggle/
        chmod 600 ~/.kaggle/kaggle.json
        echo "Kaggle credentials configured successfully"
    else
        echo "Error: kaggle.json not found in current directory"
        echo "Please place your kaggle.json file in the current directory"
        echo "You can download it from https://www.kaggle.com/settings under 'API' section"
        exit 1
    fi
}

# Create directory if it doesn't exist
ensure_dir() {
    local dir=$1
    mkdir -p "$dir"
} 