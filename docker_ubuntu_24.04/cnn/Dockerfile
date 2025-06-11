FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pandas>=2.0.0 numpy>=1.24.0 scikit-learn>=1.3.0 torch>=2.0.0 matplotlib>=3.7.0

WORKDIR /app
COPY train_cnn.py .
COPY analyze_cnn.py .
COPY ../data/intel_lab_data_cleaned.csv ./data/

CMD ["python3", "train_cnn.py", "client2"]