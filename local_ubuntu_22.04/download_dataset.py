import os
import sys
from pathlib import Path
import gdown

def download_dataset():
    """Download intel_lab_data_cleaned.csv from Google Drive."""
    print("Downloading dataset from Google Drive...")

    # Google Drive file ID
    file_id = "1zAPPGBgvBLCPLjynx0BNRFbayZSfloEp"
    dest_path = Path("data/intel_lab_data_cleaned.csv")
    url = f"https://drive.google.com/uc?id={file_id}"

    # Create data directory if it doesn't exist
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if the file already exists
    if dest_path.exists():
        print(f"Dataset already exists at {dest_path}. Skipping download.")
        return

    try:
        # Download the file
        gdown.download(url, str(dest_path), quiet=False)
    except Exception as e:
        print(f"Error: Failed to download dataset: {e}")
        sys.exit(1)

    # Verify download
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"Dataset downloaded successfully to {dest_path}")
    else:
        print("Error: Downloaded file is corrupted or too small")
        dest_path.unlink(missing_ok=True)
        sys.exit(1)

if __name__ == "__main__":
    download_dataset()