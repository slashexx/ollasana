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
  else
    echo "⚠ No NVIDIA GPU detected or nvidia-smi not available"
    echo "  Running in CPU-only mode"
    export OLLAMA_GPU=0
    export OLLAMA_GPU_LAYERS=0
  fi
  
  # Check for CUDA availability
  if [ -d "/usr/local/cuda" ] || [ -n "$CUDA_HOME" ]; then
    echo "✓ CUDA installation detected"
  else
    echo "⚠ CUDA not found - GPU acceleration may not work"
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
  
  # Get validation timeout from environment, with different defaults based on model size
  local validation_timeout=${VALIDATION_TIMEOUT:-0}
  
  # Auto-detect if this is likely a large model and set appropriate timeout
  if [ "$validation_timeout" -eq 0 ]; then
    local model_lower=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')
    # Check for large model indicators
    if echo "$model_lower" | grep -qE '(34b|70b|65b|180b|llava|vision)' || \
       ! echo "$model_lower" | grep -qE '(q4|q5|q8)'; then
      validation_timeout=1800  # 30 minutes for large models
      echo "⏳ Detected large model - using extended timeout: $validation_timeout seconds"
    else
      validation_timeout=600   # 10 minutes for smaller models
      echo "⏳ Using standard timeout: $validation_timeout seconds"
    fi
  else
    echo "⏳ Using custom timeout: $validation_timeout seconds"
  fi
  
  # First, try to preload the model for large models
  if [ "$validation_timeout" -gt 600 ]; then
    echo "🔄 Pre-loading large model (this may take several minutes)..."
    local preload_timeout=$((validation_timeout * 80 / 100))  # 80% of total timeout
    local preload_response
    preload_response=$(curl -s -X POST http://localhost:$OLLAMA_PORT/api/generate \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL_NAME\",\"prompt\":\"Hi\",\"stream\":false,\"options\":{\"num_predict\":1}}" \
      --max-time "$preload_timeout" 2>/dev/null || echo "timeout")
    
    if echo "$preload_response" | grep -q "response"; then
      echo "✓ Model pre-loading successful"
    elif [ "$preload_response" = "timeout" ]; then
      echo "⚠ Model pre-loading timed out, but continuing with validation..."
    else
      echo "⚠ Model pre-loading failed, but continuing with validation..."
    fi
  fi
  
  # Now do the actual validation
  echo "🧪 Running model validation test..."
  local test_response
  test_response=$(curl -s -X POST http://localhost:$OLLAMA_PORT/api/generate \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_NAME\",\"prompt\":\"Hello, how are you?\",\"stream\":false,\"options\":{\"num_predict\":50}}" \
    --max-time "$validation_timeout" 2>/dev/null || echo "timeout")
  
  if echo "$test_response" | grep -q "response" && [ "$test_response" != "timeout" ]; then
    echo "✓ Model validation successful"
    local sample_response=$(echo "$test_response" | grep -o '"response":"[^"]*"' | head -c 100)
    echo "  Sample response: $sample_response..."
  elif [ "$test_response" = "timeout" ]; then
    echo "✗ Model validation timed out after $validation_timeout seconds"
    echo "💡 For very large models, try setting VALIDATION_TIMEOUT environment variable to a higher value (e.g., 3600 for 1 hour)"
    exit 1
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