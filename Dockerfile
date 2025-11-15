FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Avoid interactive apt prompts
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# System packages:
#   - build-essential, cmake: build PoissonRecon/SurfaceTrimmer
#   - git: clone Pi3 & PoissonRecon
#   - libpng-dev, libjpeg-dev, zlib1g-dev: image libs used by PoissonRecon
#   - libgl1: Open3D runtime dependency on many distros
#   - libglib2.0-0, libgthread-2.0-0: OpenCV dependencies
#   - libsm6, libxext6, libfontconfig1, libxrender1: Additional OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        cmake \
        libpng-dev \
        libjpeg-dev \
        zlib1g-dev \
        libgl1 \
        libboost-dev \
        libboost-system-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libfontconfig1 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------
# Install Pi3 + Python dependencies via pip
# -------------------------------------------------------------------
WORKDIR /opt

RUN git clone https://github.com/yyfz/Pi3.git /opt/Pi3

WORKDIR /opt/Pi3

# Install Pi3's Python requirements using pip
# plus extra deps for our pipeline (open3d + safetensors).
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir open3d safetensors

# Add our pipeline script into the Pi3 repo
COPY pi3_poisson_pipeline.py /opt/Pi3/pi3_poisson_pipeline.py
COPY pipeline_config.py /opt/Pi3/pipeline_config.py

# Pre-download the Pi3 model to cache it in the image
RUN python -c "import torch; import pi3.models.pi3 as pi3_model_module; print('Pre-downloading Pi3 model from HuggingFace...'); model = pi3_model_module.Pi3.from_pretrained('yyfz233/Pi3'); print('Model downloaded and cached successfully.')"

WORKDIR /opt

# Clone and build PoissonRecon (Linux Makefile builds PoissonRecon & SurfaceTrimmer
# into PoissonRecon/Bin/Linux)
RUN git clone https://github.com/mkazhdan/PoissonRecon.git /opt/PoissonRecon \
    && cd /opt/PoissonRecon \
    && make -j16 \
    && cp Bin/Linux/PoissonRecon Bin/Linux/SurfaceTrimmer /usr/local/bin/ \
    && chmod +x /usr/local/bin/PoissonRecon /usr/local/bin/SurfaceTrimmer

# Run from inside Pi3 repo so `import pi3` just works
WORKDIR /opt/Pi3

# By default, run the pipeline script; you pass pipeline args to `docker run`
ENTRYPOINT ["python", "pi3_poisson_pipeline.py"]

