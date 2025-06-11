#!/bin/bash

echo "Setting up Docker environment for Ubuntu 24.04..."

# Install Docker
sudo apt update
sudo apt install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker

# Build and run Docker for specified model
MODEL=$1

if [ -z "$MODEL" ]; then
    echo "Please specify a model (linear_regression, random_forest, or cnn)"
    exit 1
fi

cd $MODEL
docker build -t ${MODEL} .
docker run --rm -v $(pwd)/../../ml_results_${MODEL}:/app/ml_results_${MODEL} ${MODEL}

echo "Setup and execution for ${MODEL} complete!"