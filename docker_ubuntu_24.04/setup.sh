#!/bin/bash

set -e

echo "[SETUP] Installing system dependencies..."

apt-get update && \
apt-get install -y python3 python3-pip && \
rm -rf /var/lib/apt/lists/*

echo "[SETUP] Installing Python packages..."

pip install --break-system-packages -r requirements.txt

echo "[SETUP] Done."
