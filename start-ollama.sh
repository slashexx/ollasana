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
  
  # Force GPU usage if available
  if [ "$OLLAMA_GPU" = "1" ]; then
    export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT
    export OLLAMA_KEEP_ALIVE=5m
    export OLLAMA_NUM_PARALLEL=1
    export OLLAMA_MAX_LOADED_MODELS=1
    export OLLAMA_FLASH_ATTENTION=0
    # Force GPU memory allocation
    export OLLAMA_GPU_MEMORY_FRACTION=0.8
    export OLLAMA_LLM_LIBRARY="cuda"
    echo "✓ Ollama configured to force GPU usage"
  else
    export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT
    echo "✓ Ollama configured for CPU-only mode"
  fi
  
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
  
  # For GPU setups, ensure GPU allocation during model pull
  if [ "$OLLAMA_GPU" = "1" ]; then
    echo "Configuring GPU memory for model loading..."
    export CUDA_VISIBLE_DEVICES=0
  fi
  
  if ollama pull "$MODEL_NAME"; then
    echo "✓ Successfully pulled model: $MODEL_NAME"
  else
    echo "✗ Failed to pull model: $MODEL_NAME"
    echo "Please check if the model name is correct."
    echo "Available models can be found at: https://ollama.ai/library"
    exit 1
  fi
}

# Function to force GPU usage for model
force_gpu_model_load() {
  if [ "$OLLAMA_GPU" = "1" ]; then
    echo "Forcing GPU model load..."
    
    # Test GPU allocation with a simple request
    echo "Testing GPU allocation..."
    local gpu_test_response
    gpu_test_response=$(timeout 60 curl -s -X POST http://localhost:$OLLAMA_PORT/api/generate \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL_NAME\",\"prompt\":\"Test\",\"stream\":false,\"options\":{\"num_gpu\":99}}")
    
    if echo "$gpu_test_response" | grep -q "response"; then
      echo "✓ GPU model loading test successful"
    else
      echo "⚠ GPU model loading test failed, but continuing..."
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
    --max-time 120)
  
  if echo "$test_response" | grep -q "response"; then
    echo "✓ Model validation successful"
  else
    echo "✗ Model validation failed"
    echo "Response: $test_response"
    exit 1
  fi
}

# Function to validate GPU inference is working
validate_gpu_inference() {
  if [ "$OLLAMA_GPU" = "1" ]; then
    echo "Validating GPU inference..."
    
    # Monitor GPU memory before and during inference
    local gpu_memory_before=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo "GPU memory before test: ${gpu_memory_before}MB"
    
    # Run a test inference and monitor GPU usage (with longer timeout)
    echo "Running GPU inference test (this may take a moment for first GPU load)..."
    local test_response
    test_response=$(timeout 120 curl -s -X POST http://localhost:$OLLAMA_PORT/api/generate \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL_NAME\",\"prompt\":\"Generate a short creative story about a robot.\",\"stream\":false}") &
    
    local curl_pid=$!
    
    # Wait a moment for inference to start, then check GPU memory
    sleep 10
    local gpu_memory_during=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo "GPU memory during inference: ${gpu_memory_during}MB"
    
    # Wait for the test to complete
    wait $curl_pid
    local curl_exit_code=$?
    
    # Check if GPU memory increased during inference (indicating GPU usage)
    if [ "$gpu_memory_during" -gt "$((gpu_memory_before + 100))" ]; then
      echo "✓ GPU inference validated - memory usage increased by $((gpu_memory_during - gpu_memory_before))MB"
    else
      echo "⚠ GPU memory didn't increase significantly during inference"
      echo "  Memory before: ${gpu_memory_before}MB, during: ${gpu_memory_during}MB"
      echo "  This might indicate CPU fallback - checking response..."
    fi
    
    if [ $curl_exit_code -eq 0 ] && echo "$test_response" | grep -q "response"; then
      echo "✓ Inference test completed successfully"
    elif [ $curl_exit_code -eq 124 ]; then
      echo "⚠ Inference test timed out after 120 seconds"
      echo "  This might indicate the model is loading on CPU instead of GPU"
    else
      echo "⚠ Inference test failed with exit code: $curl_exit_code"
      echo "  Response: $test_response"
    fi
  else
    echo "Skipping GPU validation (CPU-only mode)"
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

# Force GPU model load
force_gpu_model_load

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