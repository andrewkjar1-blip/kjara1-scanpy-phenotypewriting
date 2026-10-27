FROM python:3.10-slim

# System dependencies needed by scanpy/anndata
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies — replace versions with yours
RUN pip install --no-cache-dir \
    scanpy==1.9.8 \
    anndata==0.10.3 \
    pandas==2.1.0 \
    numpy==1.26.0 \
    joblib==1.3.2 \
    scipy==1.11.0

WORKDIR /pipeline
