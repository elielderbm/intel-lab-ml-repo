#!/bin/bash

set -e

IMAGE_NAME="ml-env"

# # Check if image exists
# if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
#   echo "[DOCKER] Image '$IMAGE_NAME' not found. Building..."
#   docker build -t $IMAGE_NAME ..
# else
#   echo "[DOCKER] Image '$IMAGE_NAME' already exists. Skipping build."
# fi

#docker build -t $IMAGE_NAME ..

echo "[DOCKER] Running CNN container..."

docker run -it --rm \
  -v "$(pwd)":/workspace/cnn \
  -v "$(pwd)/../data":/workspace/data \
  -v "$(pwd)/../ml_results_cnn":/workspace/ml_results_cnn \
  -w /workspace/cnn \
  $IMAGE_NAME \
  bash
