#!/bin/bash

echo "Setting up environment for Raspberry Pi 3B (Ubuntu)..."

# Update system
sudo apt update
sudo apt install -y python3 python3-pip

# Install Python dependencies
pip install --break-system-packages -r requirements.txt

# Install PyTorch for ARM (CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "Setup complete!"