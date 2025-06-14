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

echo "[DOCKER] Running Linear Regression container..."

docker run -it --rm \
  -v "$(pwd)":/workspace/linear_regression \
  -v "$(pwd)/../data":/workspace/data \
  -v "$(pwd)/../ml_results_linear_regression":/workspace/ml_results_linear_regression \
  -w /workspace/linear_regression \
  $IMAGE_NAME \
  bash
