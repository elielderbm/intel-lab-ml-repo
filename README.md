# Intel Lab ML Repository

This repository contains scripts for training and analyzing three machine learning models (**Linear Regression**, **Random Forest**, and **CNN**) on the Intel Lab Sensor Data dataset to predict temperature. The setup is designed for three environments: **Local (Ubuntu 22.04)**, **Docker (Ubuntu 24.04)**, and **Raspberry Pi 3B (Ubuntu)**.

## Repository Structure

```
intel-lab-ml-repo/
├── local_ubuntu_22.04/          # Scripts and setup for Ubuntu 22.04
├── docker_ubuntu_24.04/         # Docker setups for Ubuntu 24.04
├── raspberry_pi_ubuntu/         # Scripts and setup for Raspberry Pi
├── .gitignore                   # Git ignore file
├── README.md                    # This file
```

Each environment directory contains:
- `scripts/` or model-specific folders: Training and analysis scripts.
- `setup.sh`: Installation script for dependencies.
- `download_dataset.py`: Python script to download `intel_lab_data_cleaned.csv` from Google Drive using `gdown`.
- `requirements.txt`: Python dependencies (for Local and Raspberry Pi).
- `data/`: Destination for `intel_lab_data_cleaned.csv`.
- `ml_results_<model>/`: Output directories for results.

## Prerequisites

- **Git**: Installed on all environments (`sudo apt install git`).
- **Python 3.10+**: Required for all environments.
- **Docker**: Required for `docker_ubuntu_24.04` (Ubuntu 24.04).
- **Internet Connection**: Required to download the dataset from Google Drive.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/intel-lab-ml-repo.git
cd intel-lab-ml-repo
```

### 2. Local (Ubuntu 22.04)

1. Navigate to the environment:
   ```bash
   cd local_ubuntu_22.04
   ```

2. Install dependencies:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. Download the dataset:
   ```bash
   python3 download_dataset.py
   ```

4. Train a model (e.g., Linear Regression):
   ```bash
   python3 scripts/train_linear_regression.py client1
   ```

5. Analyze results:
   ```bash
   python3 scripts/analyze_linear_regression.py
   ```

6. Repeat for Random Forest and CNN.

### 3. Docker (Ubuntu 24.04)

1. Install Docker:
   ```bash
   sudo apt update
   sudo apt install docker.io
   sudo systemctl start docker
   sudo systemctl enable docker
   ```

2. Navigate to the environment:
   ```bash
   cd docker_ubuntu_24.04
   ```

3. Download the dataset:
   ```bash
   pip install -r ../local_ubuntu_22.04/requirements.txt
   python3 download_dataset.py
   ```

4. Run a model (e.g., CNN):
   ```bash
   chmod +x setup.sh
   ./setup.sh cnn
   ```

5. Run analysis:
   ```bash
   cd cnn
   docker run --rm -v $(pwd)/../../ml_results_cnn:/app/ml_results_cnn cnn python3 analyze_cnn.py
   ```

6. Repeat for Linear Regression and Random Forest.

### 4. Raspberry Pi 3B (Ubuntu)

1. Navigate to the environment:
   ```bash
   cd raspberry_pi_ubuntu
   ```

2. Install dependencies:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. Download the dataset:
   ```bash
   python3 download_dataset.py
   ```

4. Train a model (e.g., Random Forest):
   ```bash
   python3 scripts/train_random_forest.py client3
   ```

5. Analyze results:
   ```bash
   python3 scripts/analyze_random_forest.py
   ```

6. Repeat for Linear Regression and CNN. **Note**: For CNN, reduce `N_EPOCHS` (e.g., to 50) in `train_cnn.py` if performance is slow.

## Output

Each model generates results in `ml_results_<model>/`:
- **Training**: Models (`model_<client_id>.pkl` or `.pth`), metrics, times (`total_time_<client_id>.txt`), and plots (scatter, feature importance for Random Forest, convergence for CNN).
- **Analysis**: `analysis/` subfolder with Markdown report (`analysis_<model>.md`) and comparative plots.

## Notes

- **Dataset**: Downloaded to `data/intel_lab_data_cleaned.csv` by running `python3 download_dataset.py`. The Google Drive file ID is pre-configured in `download_dataset.py`.
- **Raspberry Pi**: May require longer download and training times, especially for CNN. PyTorch for ARM is installed via `setup.sh`.
- **Docker**: Ensure the dataset is downloaded before running `setup.sh`. Results are persisted in `ml_results_<model>/` via volume mounts.
- **Adjustments**: Modify `N_EPOCHS` (CNN) or `N_ESTIMATORS` (Random Forest) in training scripts for performance tuning.
- **Troubleshooting**:
  - If the dataset download fails, ensure the Google Drive file is set to "Anyone with the link" in sharing settings.
  - Check `download_dataset.py` output for errors.
  - If `gdown` fails, reinstall it: `pip install gdown`.

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/<name>`).
3. Commit changes (`git commit -am 'Add feature'`).
4. Push to the branch (`git push origin feature/<name>`).
5. Create a Pull Request.

## License

MIT License (see `LICENSE` file if added).