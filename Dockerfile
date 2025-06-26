# GPU-Compatible Ollama OpenAI API Server
# 
# To run with GPU support:
#   docker run --gpus all -e MODEL_NAME=llama2 -p 9000:9000 your-image
#
# To run with specific GPU:
#   docker run --gpus '"device=0"' -e MODEL_NAME=llama2 -p 9000:9000 your-image
#
# To run CPU-only:
#   docker run -e MODEL_NAME=llama2 -p 9000:9000 your-image
#
# All services are unified and accessible through port 9000
# 
# Note: GPU acceleration works via NVIDIA Container Runtime when using --gpus flag.
# CUDA doesn't need to be installed inside the container - it's provided by the runtime.

FROM ollama/ollama:latest

# Install curl, python and other utilities
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    bash \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Note: CUDA toolkit and nvidia-smi are provided by NVIDIA Container Runtime
# when using --gpus flag. The base ollama/ollama image handles GPU acceleration.

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

# Ollama specific environment variables
ENV OLLAMA_HOST=0.0.0.0
ENV OLLAMA_PORT=11434
ENV OLLAMA_MODELS=/data-models
ENV OLLAMA_HOME=/tmp/ollama_home

# GPU-specific environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=all

# Ollama GPU configuration
ENV OLLAMA_GPU=1
ENV OLLAMA_NUM_GPU=-1
ENV OLLAMA_GPU_MEMORY_FRACTION=0.9
ENV OLLAMA_MAX_LOADED_MODELS=1
ENV OLLAMA_GPU_LAYERS=999

# Additional GPU runtime environment variables
ENV NVIDIA_REQUIRE_CUDA="cuda>=11.0"

# Remove HF legacy variables - Ollama uses its own registry
# ENV HF_HOME, HF_HUB_CACHE, etc. are not needed for Ollama

# Create necessary directories
RUN mkdir -p /data-models /tmp/ollama_home

# Copy entrypoint script and API server
COPY start-ollama.sh /start-ollama.sh
COPY api_server.py /api_server.py
RUN chmod +x /start-ollama.sh
RUN chmod +x /api_server.py

# GPU runtime labels for Docker
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="12.0"

# Expose only the unified API port
EXPOSE 9000

# Health check to verify GPU availability
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:9000/health || exit 1

# Set the entrypoint
ENTRYPOINT ["/start-ollama.sh"] 