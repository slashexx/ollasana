FROM ollama/ollama:latest

# Install curl, python and other utilities
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    bash \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install requests

# Set environment variables with defaults (matching original)
ENV MODEL_NAME=""
ENV SERVED_MODEL_NAME=""
ENV PORT=9000
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

# Remove HF legacy variables - Ollama uses its own registry
# ENV HF_HOME, HF_HUB_CACHE, etc. are not needed for Ollama

# Create necessary directories
RUN mkdir -p /data-models /tmp/ollama_home

# Copy entrypoint script
COPY start-ollama.sh /start-ollama.sh
RUN chmod +x /start-ollama.sh

# Expose ports
EXPOSE 9000 11434

# Set the entrypoint
ENTRYPOINT ["/start-ollama.sh"] 