# Document Rotation Classification

A deep learning model for automatically detecting and correcting document orientation in images. The model can classify document rotations into 8 classes (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°) and provides confidence scores for its predictions.

## Features

- 🔄 8-class rotation detection (0° to 315° in 45° increments)
- 📊 Confidence scoring for predictions
- 🖼️ Support for various document types (IDs, receipts, scientific papers, etc.)
- 🚀 Efficient MobileNetV3-small backbone
- 🎯 Enhanced with CoordConv and CBAM attention mechanisms
- ⚡ PyTorch Lightning training framework
- 📈 Wandb logging and visualization
- 📦 ONNX export support

## Model Architecture

The model combines several key components for robust document rotation detection:

- **Backbone**: MobileNetV3-small for efficient feature extraction
- **Spatial Awareness**: CoordConv layers for better position-aware features
- **Attention**: CBAM (Convolutional Block Attention Module) for focusing on relevant document regions
- **Dual-head Output**:
  - Rotation classification (8 classes)
  - Confidence score prediction

## Supported Datasets

The model can be trained on multiple document datasets:

1. **RVL-CDIP**: 400,000 grayscale document images in 16 classes
2. **PubLayNet**: 360,000 scientific paper images with layout annotations
3. **MIDV-500**: Identity document images in various capture conditions
4. **SROIE**: Scanned receipt images with text annotations

## Installation

1. Clone the repository:

```bash
git clone https://github.com/dennisjooo/document-rotation.git
cd document-rotation
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download datasets:

```bash
./downloaders/download_datasets.sh
```

Note: You'll need Kaggle API credentials (kaggle.json) for downloading some datasets.

## Training

1. Configure training parameters in `configs/train.yaml`

2. Start training:

```bash
./train.sh
```

The training script will:

- Set up data loaders for the specified datasets
- Initialize the model and optimizer
- Log metrics and visualizations to Wandb (need to add WANDB_API_KEY to .env)
- Save checkpoints periodically

## Acknowledgments

This project uses the following datasets:

- RVL-CDIP (Harley et al., ICDAR 2015)
- PubLayNet (Zhong et al., ICDAR 2019)
- MIDV-500 (Bulatov et al., 2020)
- SROIE (Huang et al., ICDAR 2019)
