# Dockerfile for Hetero-KVCache-Optimizer Artifact Evaluation
# Base image: NVIDIA CUDA 12.1 with Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_VERBOSITY=error
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /artifact

# Copy dependency manifest first for layer caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel \
    && pip3 install --no-cache-dir -r requirements.txt

# Copy entire artifact
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 aeuser && chown -R aeuser:aeuser /artifact
USER aeuser

# Default command: run the full experiment pipeline
CMD ["bash", "run_all_experiments.sh"]
