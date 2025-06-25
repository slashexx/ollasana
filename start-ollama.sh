#!/bin/bash
set -e

MODEL_NAME=$(echo "$MODEL_NAME" | xargs)

# Validate that MODEL_NAME is provided
if [ -z "$MODEL_NAME" ]; then
  echo "ERROR: MODEL_NAME environment variable is required"
  echo "Please provide an Ollama model name (e.g., llama2, mistral, codellama, etc.)"
  echo "See https://ollama.ai/library for available models"
  exit 1
fi

# Determine SERVED_MODEL_NAME
if [ -z "$SERVED_MODEL_NAME" ]; then
  SERVED_MODEL_NAME="$MODEL_NAME"
  echo "SERVED_MODEL_NAME not provided, using: $SERVED_MODEL_NAME"
fi

# Ensure we have a writable ollama models directory
mkdir -p "$OLLAMA_MODELS"
mkdir -p "$OLLAMA_HOME"

echo "=== Ollama Model Server Configuration ==="
echo "Model: $MODEL_NAME"
echo "Served Model Name: $SERVED_MODEL_NAME"
echo "Unified API Port: ${PORT:-9000}"
echo "Internal Ollama Port: $OLLAMA_PORT (not exposed)"
echo "Models Directory: $OLLAMA_MODELS"
echo "========================================="

# Function to detect and configure GPU
detect_gpu() {
  echo "=== GPU Detection ==="
  
  # Check for NVIDIA GPU
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>/dev/null || echo "Could not query GPU details"
    
    # Get GPU memory for aggressive allocation (especially for 70B models)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "GPU Memory: ${GPU_MEMORY}MB"
    
    # Set aggressive GPU-specific Ollama environment variables for large models
    export OLLAMA_GPU_LAYERS=-1  # Use all available GPU layers (-1 = auto-detect max)
    export OLLAMA_GPU=1
    export OLLAMA_NUM_GPU=-1     # Use all available GPUs
    export OLLAMA_GPU_MEMORY_FRACTION=0.95  # Use 95% of GPU memory
    export OLLAMA_MAX_LOADED_MODELS=1       # Only load one model at a time for memory efficiency
    export OLLAMA_MAX_QUEUE=4               # Allow some queuing for concurrent requests
    
    # Set CUDA environment variables for better performance
    export CUDA_VISIBLE_DEVICES=all
    export CUDA_CACHE_DISABLE=0
    export CUDA_CACHE_MAXSIZE=2147483648
    
    echo "✓ Ollama configured for GPU acceleration with aggressive settings for large models"
  else
    echo "⚠ No NVIDIA GPU detected or nvidia-smi not available"
    echo "  Running in CPU-only mode"
    export OLLAMA_GPU=0
    export OLLAMA_GPU_LAYERS=0
  fi
  
  # Check for CUDA availability
  if [ -d "/usr/local/cuda" ] || [ -n "$CUDA_HOME" ] || command -v nvcc >/dev/null 2>&1; then
    echo "✓ CUDA installation detected"
    # Additional CUDA runtime check
    if command -v nvcc >/dev/null 2>&1; then
      CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
      echo "CUDA Version: $CUDA_VERSION"
    fi
  else
    echo "⚠ CUDA not found - GPU acceleration may not work"
    echo "  Installing minimal CUDA runtime..."
    # Try to ensure CUDA libraries are available
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
    export PATH="/usr/local/cuda/bin:${PATH}"
  fi
  
  echo "========================"
}

# Function to start Ollama server in background
start_ollama_server() {
  echo "Starting internal Ollama server on localhost:$OLLAMA_PORT..."
  export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT
  ollama serve &
  OLLAMA_PID=$!
  
  # Wait for Ollama server to be ready
  echo "Waiting for Ollama server to be ready..."
  for i in {1..60}; do
    if curl -s http://localhost:$OLLAMA_PORT/api/tags >/dev/null 2>&1; then
      echo "✓ Ollama server is ready"
      return 0
    fi
    if [ $i -eq 60 ]; then
      echo "✗ Ollama server failed to start within 60 seconds"
      exit 1
    fi
    echo "  Waiting... ($i/60)"
    sleep 1
  done
}

# Function to pull/load Ollama model
load_ollama_model() {
  echo "Loading Ollama model: $MODEL_NAME"
  
  # Check if model is already available locally
  if ollama list | grep -q "^$MODEL_NAME"; then
    echo "✓ Model $MODEL_NAME is already available locally"
    return 0
  fi
  
  echo "Pulling model $MODEL_NAME from Ollama registry..."
  echo "This may take several minutes depending on model size..."
  
  if ollama pull "$MODEL_NAME"; then
    echo "✓ Successfully pulled model: $MODEL_NAME"
  else
    echo "✗ Failed to pull model: $MODEL_NAME"
    echo "Please check if the model name is correct."
    echo "Available models can be found at: https://ollama.ai/library"
    exit 1
  fi
}

# Function to validate model is working
validate_model() {
  echo "Validating model functionality..."
  
  # Test the model with a simple prompt
  local test_response
  test_response=$(curl -s -X POST http://localhost:$OLLAMA_PORT/api/generate \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_NAME\",\"prompt\":\"Hello\",\"stream\":false}" \
    --max-time 30)
  
  if echo "$test_response" | grep -q "response"; then
    echo "✓ Model validation successful"
  else
    echo "✗ Model validation failed"
    echo "Response: $test_response"
    exit 1
  fi
}

# Function to start FastAPI server
start_fastapi_server() {
  echo "Starting FastAPI server..."
  python3 /api_server.py
}

# Function to cleanup on exit
cleanup() {
  echo "Shutting down services..."
  if [ ! -z "$OLLAMA_PID" ]; then
    kill $OLLAMA_PID 2>/dev/null || true
  fi
  exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Main execution flow
echo "🦙 Starting Ollama-based OpenAI-compatible API server"

# Detect and configure GPU
detect_gpu

# Start Ollama server
start_ollama_server

# Load the Ollama model
load_ollama_model

# Validate model is working
validate_model

echo "✅ All services started successfully!"
echo ""
echo "🌐 Unified OpenAI-Compatible API Server"
echo "Model: $MODEL_NAME"
echo "Public API Endpoint: http://0.0.0.0:${PORT:-9000}"
echo "Internal Ollama Server: http://localhost:$OLLAMA_PORT (internal only)"
echo ""
echo "📋 Available endpoints:"
echo "   GET  /v1/models"
echo "   GET  /health"
echo "   POST /v1/chat/completions"
echo "   POST /v1/completions"
echo "   POST /v1/embeddings"
echo "   POST /v1/moderations"
echo "   + Additional OpenAI-compatible endpoints"
echo ""
echo "🔗 Access your API at: http://localhost:${PORT:-9000}"
echo ""

# Start the FastAPI server (runs in foreground)
start_fastapi_server 