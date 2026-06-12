FROM --platform=linux/amd64 nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# System deps — Python, GDAL (3.4.x on Ubuntu 22.04), JP2 codec
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip \
        gdal-bin libgdal-dev python3-gdal \
        libopenjp2-7 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch (CUDA 11.8 wheels) + remaining deps — skip GDAL (installed via apt)
COPY requirements.txt .
RUN pip install --no-cache-dir \
        torch==2.1.0 torchvision==0.16.0 \
        --index-url https://download.pytorch.org/whl/cu118 && \
    grep -vE '^(GDAL|torch)' requirements.txt | pip install --no-cache-dir -r /dev/stdin

# Source code + bundled pypvroof data
COPY . .

ENTRYPOINT ["python3", "main.py"]
