# JetPack 6.2 userland on x86_64 (Ubuntu 22.04)
# CUDA 12.6.x + cuDNN 9.3.0 + TensorRT 10.3.0 + optional VPI 3.x
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDNN_DEB_VER=9.3.0.75-1
# Put the TensorRT tarball in the build context with this exact filename
# (x86_64 build compiled for CUDA 12.5; works fine with CUDA 12.6 userland)
# get it from NVIDIA Developer (login required):
# TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.5.tar.gz
ARG TRT_TAR=TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.5.tar.gz

# Basics
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates gnupg2 software-properties-common wget curl \
    build-essential cmake git pkg-config \
    python3.10 python3-pip python3-dev \
    blender libsm6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Add NVIDIA CUDA apt repo + pin (so we can pin cuDNN 9.3 exactly)
RUN wget -qO /etc/apt/preferences.d/cuda-repository-pin-600 \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin \
    && apt-key adv --fetch-keys \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
    && add-apt-repository \
    "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    libcudnn9-cuda-12=${CUDNN_DEB_VER} \
    libcudnn9-dev-cuda-12=${CUDNN_DEB_VER} \
    && rm -rf /var/lib/apt/lists/*

# --- TensorRT 10.3.0 (from official tarball) ---
# Copy the tarball you downloaded into the build context before building.
COPY ${TRT_TAR} /tmp/${TRT_TAR}
RUN mkdir -p /opt/tensorrt && \
    tar -xzf /tmp/${TRT_TAR} -C /opt/tensorrt --strip-components=1 && \
    rm /tmp/${TRT_TAR}
# Install Python wheel(s) shipped in the tarball (match cp310 for Ubuntu 22.04)
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install numpy && \
    python3 -m pip install /opt/tensorrt/python/*cp310*.whl || true
ENV TENSORRT_DIR=/opt/tensorrt
ENV LD_LIBRARY_PATH=${TENSORRT_DIR}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}

# ----- Workspace: Deep_Object_Pose -----
# Keep dependencies baked into the image but mount the workspace at runtime (see docker-compose.yml).
WORKDIR /workspace
COPY Deep_Object_Pose/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    python3 -m pip install --no-cache-dir blenderproc pyquaternion && \
    rm /tmp/requirements.txt
RUN blenderproc pip install pyquaternion "numpy>=1.25.2,<2.1"

# Expect the host repository to be bind-mounted at /workspace/Deep_Object_Pose when the container runs.
WORKDIR /workspace/Deep_Object_Pose
VOLUME ["/workspace/Deep_Object_Pose"]

# --- OPTIONAL: VPI 3.x host libs for Jammy (matches r36.x era) ---
# Uncomment to add NVIDIA Jetson host repo for VPI 3 on x86_64 Jammy and install it.
# RUN apt-get update && apt-get install -y gnupg && \
#     apt-key adv --fetch-keys https://repo.download.nvidia.com/jetson/jetson-ota-public.asc && \
#     add-apt-repository "deb https://repo.download.nvidia.com/jetson/x86_64/jammy r36.2 main" && \
#     apt-get update && apt-get install -y --no-install-recommends \
#         libnvvpi3 vpi3-dev vpi3-samples python3.10-vpi3 && \
#     rm -rf /var/lib/apt/lists/*

# Dev niceties & sanity checks
RUN python3 -m pip install --no-cache-dir polygraphy==0.49.9 onnx==1.16.1 && \
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /etc/bash.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/tensorrt/lib:$LD_LIBRARY_PATH' >> /etc/bash.bashrc

# Print versions when the container starts (handy for audit)
CMD bash -lc '\
    echo "CUDA:" && nvcc --version; \
    echo "\ncuDNN:" && ldconfig -p | grep -E "libcudnn(.*)so" | head -n 3; \
    echo "\nTensorRT libs:" && ls -1 /opt/tensorrt/lib/libnvinfer.so*; \
    python3 - <<PY \
    import sys; \
    print("\\nPython:", sys.version); \
    try: import tensorrt as trt; print("TensorRT Python:", trt.__version__); \
    except Exception as e: print("TensorRT Python not found:", e) \
    PY'
