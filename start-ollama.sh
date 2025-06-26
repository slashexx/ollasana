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
    
    # Set GPU-specific Ollama environment variables
    export OLLAMA_GPU_LAYERS=999  # Use all GPU layers
    export OLLAMA_GPU=1
    echo "✓ Ollama configured for GPU acceleration"
    
    # In containerized environments, check for GPU runtime availability instead of CUDA installation
    if [ -n "$NVIDIA_VISIBLE_DEVICES" ] || [ -n "$CUDA_VISIBLE_DEVICES" ]; then
      echo "✓ NVIDIA Container Runtime detected - GPU acceleration enabled"
    elif [ -d "/usr/local/cuda" ] || [ -n "$CUDA_HOME" ]; then
      echo "✓ CUDA installation detected"
    else
      echo "ℹ Running in containerized environment - GPU access via NVIDIA Container Runtime"
    fi
    
    # Verify GPU memory is accessible
    if nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits >/dev/null 2>&1; then
      local gpu_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
      echo "✓ GPU memory accessible: ${gpu_memory}MB free"
    fi
    
    # Add diagnostic information for troubleshooting
    echo "Diagnostic info:"
    echo "  - Driver version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits)"
    echo "  - CUDA Version: $(nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "Unknown")"
    echo "  - Container GPU access: ${NVIDIA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-"not set"}}"
    
  else
    echo "⚠ No NVIDIA GPU detected or nvidia-smi not available"
    echo "  Running in CPU-only mode"
    export OLLAMA_GPU=0
    export OLLAMA_GPU_LAYERS=0
  fi
  
  echo "========================"
}

# Function to start Ollama server in background
start_ollama_server() {
  echo "Starting internal Ollama server on localhost:$OLLAMA_PORT..."
  
  # Configure Ollama for GPU usage with safer settings
  if [ "$OLLAMA_GPU" = "1" ]; then
    export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT
    export OLLAMA_KEEP_ALIVE=5m
    export OLLAMA_NUM_PARALLEL=1
    export OLLAMA_MAX_LOADED_MODELS=1
    export OLLAMA_FLASH_ATTENTION=0
    # Explicit GPU configuration
    export CUDA_VISIBLE_DEVICES=0
    export OLLAMA_GPU_MEMORY_FRACTION=0.9
    export OLLAMA_NUM_GPU=1
    echo "✓ Ollama configured for GPU usage (safe mode)"
    echo "  GPU Device: ${CUDA_VISIBLE_DEVICES}"
    echo "  GPU Layers: ${OLLAMA_GPU_LAYERS}"
    
    # Try starting with GPU first
    echo "Attempting to start Ollama with GPU support..."
    ollama serve &
    OLLAMA_PID=$!
    
    # Wait a moment to see if it crashes
    sleep 5
    if ! kill -0 $OLLAMA_PID 2>/dev/null; then
      echo "⚠ GPU initialization failed, falling back to CPU mode..."
      export OLLAMA_GPU=0
      export OLLAMA_GPU_LAYERS=0
      unset OLLAMA_GPU_MEMORY_FRACTION
      unset CUDA_VISIBLE_DEVICES
      export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT
      ollama serve &
      OLLAMA_PID=$!
      echo "✓ Ollama restarted in CPU-only mode"
    else
      echo "✓ Ollama started successfully with GPU support"
    fi
  else
    export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT
    ollama serve &
    OLLAMA_PID=$!
    echo "✓ Ollama configured for CPU-only mode"
  fi
  
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
  else
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
  fi
  
  # Force GPU loading if available
  if [ "$OLLAMA_GPU" = "1" ]; then
    echo "Forcing GPU model load for first inference..."
    local gpu_load_response
    gpu_load_response=$(curl -s -X POST http://localhost:$OLLAMA_PORT/api/generate \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL_NAME\",\"prompt\":\"GPU test\",\"stream\":false,\"options\":{\"num_gpu\":99,\"main_gpu\":0}}" \
      --max-time 30)
    
    if echo "$gpu_load_response" | grep -q "response"; then
      echo "✓ GPU model loading successful"
    else
      echo "⚠ GPU model loading failed, but model is available"
    fi
  fi
}

# Function to validate model is working
validate_model() {
  echo "Validating model functionality..."
  
  # Test the model with a simple prompt (increased timeout for GPU model loading)
  local test_response
  test_response=$(curl -s -X POST http://localhost:$OLLAMA_PORT/api/generate \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_NAME\",\"prompt\":\"Hello\",\"stream\":false}" \
    --max-time 60)
  
  if echo "$test_response" | grep -q "response"; then
    echo "✓ Model validation successful"
  else
    echo "✗ Model validation failed"
    echo "Response: $test_response"
    exit 1
  fi
}

# Function to validate GPU inference is working (simplified)
validate_gpu_inference() {
  if [ "$OLLAMA_GPU" = "1" ]; then
    echo "Checking GPU usage during inference..."
    
    # Monitor GPU memory before inference
    local gpu_memory_before=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo "GPU memory before: ${gpu_memory_before}MB"
    
    # Wait a moment then check again
    sleep 5
    local gpu_memory_after=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo "GPU memory after model load: ${gpu_memory_after}MB"
    
    # If there's significant GPU memory usage, GPU is working
    if [ "$gpu_memory_after" -gt "$((gpu_memory_before + 500))" ]; then
      echo "✓ GPU inference appears to be working (${gpu_memory_after}MB in use)"
    else
      echo "ℹ GPU memory usage: ${gpu_memory_after}MB (may be using CPU or small model)"
    fi
  else
    echo "Running in CPU-only mode"
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

# Validate GPU inference is working
validate_gpu_inference

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