#!/bin/bash

# Load environment variables from .env if it exists
if [ -f ".env" ]; then
    echo "Loading environment variables from .env..."
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Warning: .env file not found!"
    echo "Please create a .env file with your WANDB_API_KEY"
    echo "Example:"
    echo "WANDB_API_KEY=your_api_key_here"
    exit 1
fi

# Check for WANDB_API_KEY
if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY not found in .env!"
    echo "Please add your Weights & Biases API key to .env"
    echo "You can find it at https://wandb.ai/settings"
    exit 1
fi

# Create Python virtual environment if it doesn't exist
if [ ! -d "env" ]; then
    echo "Creating Python virtual environment..."
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    
    # Install additional dependencies for dataset downloading
    pip install kaggle python-dotenv
    sudo apt-get update && sudo apt-get install -y git-lfs
    
    # Install OpenCV system dependencies
    echo "Installing OpenCV system dependencies..."
    sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 libxext-dev
else
    echo "Python environment exists, activating..."
    source env/bin/activate
fi

# Check for kaggle.json
if [ ! -f "kaggle.json" ]; then
    echo "Error: kaggle.json not found!"
    echo "Please place your kaggle.json file in the project root directory"
    echo "You can download it from https://www.kaggle.com/settings under 'API' section"
    exit 1
fi

# Setup Kaggle credentials
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download the datasets
echo "Downloading datasets..."
bash downloaders/download_datasets.sh

# Create a unique session name using timestamp
SESSION_NAME="doc_rotation_$(date +%Y%m%d_%H%M%S)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Store the log file path
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"

# Start a new tmux session
tmux new-session -d -s "$SESSION_NAME"

# Send commands to the tmux session
tmux send-keys -t "$SESSION_NAME" "source env/bin/activate" C-m
tmux send-keys -t "$SESSION_NAME" "export WANDB_API_KEY=$WANDB_API_KEY" C-m  # Set wandb key in tmux session
tmux send-keys -t "$SESSION_NAME" "echo 'Starting training at $(date)'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'CUDA devices available:'" C-m
tmux send-keys -t "$SESSION_NAME" "python -c 'import torch; print(torch.cuda.device_count())'" C-m
tmux send-keys -t "$SESSION_NAME" "nvidia-smi" C-m

# Run training with output logged
tmux send-keys -t "$SESSION_NAME" "python train.py > '$LOG_FILE' 2>&1" C-m

# Print helpful information
echo "Starting training in tmux session: $SESSION_NAME"
echo "Training log: $LOG_FILE"
echo ""
echo "Tmux commands:"
echo "- To detach from session: press Ctrl+B, then D"
echo "- To reattach later: tmux attach-session -t $SESSION_NAME"
echo "- To list all sessions: tmux ls"
echo "- To kill the session: tmux kill-session -t $SESSION_NAME"
echo ""
echo "Monitor training:"
echo "- View logs: tail -f $LOG_FILE"
echo "- Monitor GPU: watch -n 1 nvidia-smi"
echo "- View training progress: https://wandb.ai"
echo ""

# Attach to the tmux session
tmux attach-session -t "$SESSION_NAME" 