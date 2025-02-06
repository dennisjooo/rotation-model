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

# Source utilities and dataset downloaders
source "$SCRIPT_DIR/utils.sh"
source "$SCRIPT_DIR/datasets/rvl_cdip.sh"
source "$SCRIPT_DIR/datasets/publaynet.sh"
source "$SCRIPT_DIR/datasets/midv500.sh"
source "$SCRIPT_DIR/datasets/sroie.sh"
source "$SCRIPT_DIR/datasets/docbank.sh"
source "$SCRIPT_DIR/datasets/chartqa.sh"
source "$SCRIPT_DIR/datasets/plotqa.sh"
source "$SCRIPT_DIR/datasets/cord.sh"

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