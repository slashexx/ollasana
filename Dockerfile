# GPU-Only Ollama OpenAI API Server
# 
# REQUIRES GPU - No CPU fallback!
#
# To run with GPU support:
#   docker run --gpus all -e MODEL_NAME=llama2 -p 9000:9000 your-image
#
# To run with specific GPU:
#   docker run --gpus '"device=0"' -e MODEL_NAME=llama2 -p 9000:9000 your-image
#
# All services are unified and accessible through port 9000

FROM ollama/ollama:latest

# Install curl, python and other utilities
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    bash \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Note: nvidia-smi and GPU drivers will be mounted at runtime by Docker
# when using --gpus flag. No need to install nvidia-container-toolkit inside container.

# Install Python dependencies
RUN pip3 install requests fastapi uvicorn

# Set environment variables with defaults (matching original)
ENV MODEL_NAME=""
ENV SERVED_MODEL_NAME=""
ENV PORT=9000
# Global unified port - all API traffic goes through port 9000
ENV MAX_MODEL_LEN=8192
ENV QUANTIZATION=""
ENV AWQ_WEIGHTS_PATH=""
ENV GGUF_MODEL_PATH=""
ENV TENSOR_PARALLEL_SIZE=""
ENV GPU_MEMORY_UTILIZATION="NAN"
ENV API_KEY=""
ENV SWAP_SPACE="NAN"
ENV ENABLE_STREAMING=""
ENV BLOCK_SIZE="NAN"
ENV VALIDATION_TIMEOUT="0"

# Ollama specific environment variables
ENV OLLAMA_HOST=0.0.0.0
ENV OLLAMA_PORT=11434
ENV OLLAMA_MODELS=/data-models
ENV OLLAMA_HOME=/tmp/ollama_home

# GPU-specific environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=all

# Ollama GPU configuration - Force GPU usage for large models
ENV OLLAMA_GPU=1
ENV OLLAMA_NUM_GPU=-1
ENV OLLAMA_GPU_MEMORY_FRACTION=0.9
ENV OLLAMA_MAX_LOADED_MODELS=1
ENV OLLAMA_LOAD_TIMEOUT=1800

# Force GPU acceleration even if CUDA detection is wonky
ENV OLLAMA_NOHISTORY=1
ENV OLLAMA_LLM_LIBRARY=cuda_v12

# Remove HF legacy variables - Ollama uses its own registry
# ENV HF_HOME, HF_HUB_CACHE, etc. are not needed for Ollama

# Create necessary directories
RUN mkdir -p /data-models /tmp/ollama_home

# Copy unified server
COPY main.py /main.py
RUN chmod +x /main.py

# GPU runtime labels for Docker
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="12.0"

# Expose only the unified API port
EXPOSE 9000

# Health check to verify GPU availability
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:9000/health || exit 1

# Set the entrypoint
ENTRYPOINT ["python3", "/main.py"] 