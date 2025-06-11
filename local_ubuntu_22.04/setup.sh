#!/bin/bash

echo "Setting up environment for Ubuntu 22.04..."

# Update system
sudo apt update
sudo apt install -y python3 python3-pip

# Install Python dependencies
pip install -r requirements.txt

echo "Setup complete!"