#!/bin/bash

# ==========================================================================
# Document Dataset Downloader
# ==========================================================================
# This script downloads and organizes multiple document datasets:
# - RVL-CDIP: Document classification dataset (16 classes, ~400k images)
# - PubLayNet: Document layout analysis dataset (360k document images)
# - MIDV-500: Identity document dataset (50 types of ID documents)
# - SROIE: Receipt information extraction dataset (626 receipt images)
# - DocBank: Academic document dataset with fine-grained token labels
# - Chart-QA: Dataset of charts and visualizations
# - PlotQA: Scientific plots dataset
# - CORD: Credit card OCR dataset
#
# Dataset Details:
# 1. RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing)
#    - 16 document classes (letter, form, email, handwritten, etc.)
#    - ~400,000 grayscale images
#    - Source: https://www.kaggle.com/datasets/uditamin/rvl-cdip-small
#
# 2. PubLayNet
#    - Document layout analysis with 5 classes
#    - 360,000 document images with annotations
#    - Source: https://huggingface.co/datasets/lhoestq/small-publaynet-wds
#
# 3. MIDV-500
#    - 50 different types of ID documents
#    - Multiple capture conditions (angles, lighting, etc.)
#    - Source: ftp://smartengines.com/midv-500/
#
# 4. SROIE (Scanned Receipts OCR and Information Extraction)
#    - 626 receipt images with text annotations
#    - Task: OCR and key information extraction
#    - Source: https://www.kaggle.com/datasets/urbikn/sroie-datasetv2
#
# 5. DocBank
#    - 1000+ academic documents with text annotations
#    - Task: OCR and key information extraction
#    - Source: https://github.com/doc-analysis/DocBank
#
# 6. Chart-QA
#    - 1000+ scientific charts with text annotations
#    - Task: OCR and key information extraction
#    - Source: https://huggingface.co/datasets/HuggingFaceM4/ChartQA
#
# 7. PlotQA
#    - 1000+ scientific plots with text annotations
#    - Task: OCR and key information extraction
#    - Source: https://github.com/NiteshMethani/PlotQA
#
# 8. CORD
#    - 1000+ credit cards with text annotations
#    - Task: OCR and key information extraction
#    - Source: https://huggingface.co/datasets/naver-clova-ix/cord-v2
#
# Requirements:
# - Kaggle API credentials (kaggle.json)
# - Python with pip
# - git and git-lfs
# - Internet connection
#
# Usage:
#   ./downloaders/download_datasets.sh
#
# The script will:
# 1. Install required tools (Kaggle CLI, git-lfs)
# 2. Set up Kaggle credentials
# 3. Create data directories
# 4. Download and extract datasets
# 5. Clean up temporary files
# ==========================================================================

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$(dirname "$SCRIPT_DIR")"  # Move to parent directory of script

# ========== Helper Functions ========== #
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

download_rvl_cdip() {
    if [ -d "data/rvl-cdip" ] && [ "$(ls -A data/rvl-cdip/*.tif 2>/dev/null)" ]; then
        echo "RVL-CDIP dataset already exists, skipping download..."
        return
    fi
    echo "Downloading RVL-CDIP from Kaggle..."
    mkdir -p data/rvl-cdip
    kaggle datasets download uditamin/rvl-cdip-small -p data/rvl-cdip
    
    echo "Extracting dataset..."
    unzip -j data/rvl-cdip/rvl-cdip-small.zip "data/**/*.tif" -d data/rvl-cdip
    
    echo "Cleaning up..."
    rm data/rvl-cdip/rvl-cdip-small.zip
}

download_publaynet() {
    if [ -d "data/publaynet" ] && [ "$(ls -A data/publaynet/*.png 2>/dev/null)" ]; then
        echo "PubLayNet dataset already exists and is extracted, skipping..."
        return
    fi
    
    echo "Downloading PubLayNet from HuggingFace..."
    mkdir -p data/publaynet
    git clone https://huggingface.co/datasets/lhoestq/small-publaynet-wds data/publaynet-temp
    mv data/publaynet-temp/publaynet-train-*.tar data/publaynet/ || {
        echo "Error: No publaynet tar files found in the downloaded repository"
        exit 1
    }
    rm -rf data/publaynet-temp
    
    echo "Extracting PubLayNet tar files..."
    for tarfile in data/publaynet/*.tar; do
        tar xf "$tarfile" -C data/publaynet/
        rm "$tarfile"
    done

    echo "Cleaning up non-image files..."
    find data/publaynet -type f ! -name "*.png" -delete
}

download_midv500() {
    if [ -d "data/midv500" ] && [ "$(ls -A data/midv500/*.tif 2>/dev/null)" ]; then
        echo "MIDV-500 dataset already exists, skipping download..."
        return
    fi
    echo "Downloading MIDV-500 using Python script..."
    mkdir -p data/midv500/temp
    python "$SCRIPT_DIR/download_midv.py" --dir data/midv500/temp --yes
    
    echo "Moving all TIFF images to root directory..."
    find data/midv500/temp -type f -name "*.tif" -exec mv {} data/midv500/ \;
    
    echo "Cleaning up..."
    rm -rf data/midv500/temp
}

download_sroie() {
    if [ -d "data/sroie" ] && [ "$(ls -A data/sroie/*.jpg 2>/dev/null)" ]; then
        echo "SROIE dataset already exists, skipping download..."
        return
    fi
    echo "Downloading SROIE from Kaggle..."
    mkdir -p data/sroie/temp
    kaggle datasets download urbikn/sroie-datasetv2 -p data/sroie/temp
    
    echo "Extracting dataset..."
    unzip data/sroie/temp/sroie-datasetv2.zip -d data/sroie/temp
    
    echo "Moving all images to root directory..."
    find data/sroie/temp/SROIE2019 -type f -path "*/img/*.jpg" -exec mv {} data/sroie/ \;
    
    echo "Cleaning up..."
    rm -rf data/sroie/temp
}

download_docbank() {
    if [ -d "data/docbank" ] && [ "$(ls -A data/docbank/*.pdf 2>/dev/null)" ]; then
        echo "DocBank dataset already exists, skipping download..."
        return
    fi
    echo "Downloading DocBank dataset..."
    mkdir -p data/docbank
    
    # Download from official source
    wget https://github.com/doc-analysis/DocBank/raw/master/DocBank_samples/DocBank_samples.zip -P data/docbank/
    unzip data/docbank/DocBank_samples.zip -d data/docbank/
    rm data/docbank/DocBank_samples.zip
}

download_chartqa() {
    if [ -d "data/chartqa" ] && [ "$(ls -A data/chartqa/*.jpg 2>/dev/null)" ]; then
        echo "Chart-QA dataset already exists, skipping download..."
        return
    fi
    echo "Downloading Chart-QA dataset..."
    mkdir -p data/chartqa
    
    # Download from HuggingFace
    git clone https://huggingface.co/datasets/HuggingFaceM4/ChartQA data/chartqa-temp
    mv data/chartqa-temp/train/* data/chartqa/
    rm -rf data/chartqa-temp
}

download_plotqa() {
    if [ -d "data/plotqa" ] && [ "$(ls -A data/plotqa/*.png 2>/dev/null)" ]; then
        echo "PlotQA dataset already exists, skipping download..."
        return
    fi
    echo "Downloading PlotQA dataset..."
    mkdir -p data/plotqa
    
    # Download from Google Drive using gdown
    echo "Downloading PlotQA dataset from Google Drive..."
    gdown https://drive.google.com/uc?id=1i74NRCEb-x44xqzAovuglex5d583qeiF -O data/plotqa/png.tar.gz
    
    echo "Extracting dataset..."
    tar -xzf data/plotqa/png.tar.gz -C data/plotqa/
    rm data/plotqa/png.tar.gz
    
    # Move all PNG images to root directory if they're in subdirectories
    find data/plotqa -type f -name "*.png" -exec mv {} data/plotqa/ \;
    
    # Clean up any empty subdirectories
    find data/plotqa -type d -empty -delete
}

download_cord() {
    if [ -d "data/cord" ] && [ "$(ls -A data/cord/*.jpg 2>/dev/null)" ]; then
        echo "CORD dataset already exists, skipping download..."
        return
    fi
    echo "Downloading CORD dataset..."
    mkdir -p data/cord
    
    # Download from HuggingFace
    git clone https://huggingface.co/datasets/cord-v2 data/cord-temp
    mv data/cord-temp/train/* data/cord/
    rm -rf data/cord-temp
}

# ========== Main Script ========== #
# Create base data directories
mkdir -p data/{rvl-cdip,publaynet,midv500,sroie,docbank,chartqa,plotqa,cord}

# Check and install dependencies
check_dependencies

# Setup Kaggle authentication
setup_kaggle_auth

echo "Starting downloads (will skip if datasets already exist)..."

# Download datasets
download_rvl_cdip
download_publaynet
download_midv500
download_sroie
download_docbank
download_chartqa
download_plotqa
download_cord

echo "Done! Datasets have been downloaded and organized in the data directory."

# Show the final directory structure
echo "Dataset structure:"
tree data -L 2 