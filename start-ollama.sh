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

# Function to create OpenAI-compatible API proxy
create_api_proxy() {
  cat > /tmp/api_proxy.py << 'EOF'
#!/usr/bin/env python3
import json
import requests
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import time
import os

class OllamaProxyHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.api_key = os.environ.get('API_KEY', '')
        self.served_model_name = os.environ.get('SERVED_MODEL_NAME', '')
        self.model_name = os.environ.get('MODEL_NAME', '')
        self.ollama_port = os.environ.get('OLLAMA_PORT', '11434')
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        # Suppress default HTTP server logs
        pass
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
    
    def do_GET(self):
        if self.path == '/v1/models':
            self.handle_models()
        elif self.path == '/health' or self.path == '/v1/health':
            self.handle_health()
        elif self.path.startswith('/v1/files'):
            self.handle_files_get()
        elif self.path.startswith('/v1/fine-tuning/jobs'):
            self.handle_fine_tuning_get()
        elif self.path.startswith('/v1/assistants'):
            self.handle_assistants_get()
        elif self.path.startswith('/v1/threads'):
            self.handle_threads_get()
        elif self.path.startswith('/v1/messages'):
            self.handle_messages_get()
        elif self.path.startswith('/v1/runs'):
            self.handle_runs_get()
        else:
            self.send_error(404, 'Endpoint not found')
    
    def do_POST(self):
        if self.path == '/v1/chat/completions':
            self.handle_chat_completions()
        elif self.path == '/v1/completions':
            self.handle_completions()
        elif self.path == '/v1/embeddings':
            self.handle_embeddings()
        elif self.path == '/v1/moderations':
            self.handle_moderations()
        elif self.path == '/v1/images/generations':
            self.handle_images_generations()
        elif self.path == '/v1/audio/transcriptions':
            self.handle_audio_transcriptions()
        elif self.path == '/v1/audio/translations':
            self.handle_audio_translations()
        elif self.path == '/v1/files':
            self.handle_files_post()
        elif self.path.startswith('/v1/fine-tuning/jobs'):
            self.handle_fine_tuning_post()
        elif self.path.startswith('/v1/assistants'):
            self.handle_assistants_post()
        elif self.path.startswith('/v1/threads'):
            self.handle_threads_post()
        elif self.path.startswith('/v1/messages'):
            self.handle_messages_post()
        elif self.path.startswith('/v1/runs'):
            self.handle_runs_post()
        else:
            self.send_error(404, 'Endpoint not found')
    
    def do_DELETE(self):
        if self.path.startswith('/v1/files/'):
            self.handle_files_delete()
        elif self.path.startswith('/v1/fine-tuning/jobs/'):
            self.handle_fine_tuning_delete()
        elif self.path.startswith('/v1/assistants/'):
            self.handle_assistants_delete()
        elif self.path.startswith('/v1/threads/'):
            self.handle_threads_delete()
        else:
            self.send_error(404, 'Endpoint not found')
    
    def do_PATCH(self):
        if self.path.startswith('/v1/assistants/'):
            self.handle_assistants_patch()
        elif self.path.startswith('/v1/threads/'):
            self.handle_threads_patch()
        else:
            self.send_error(404, 'Endpoint not found')
    
    def check_auth(self):
        if not self.api_key:
            return True
        
        auth_header = self.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            return token == self.api_key
        return False
    
    def handle_models(self):
        if not self.check_auth():
            self.send_error(401, 'Unauthorized')
            return
        
        models_data = {
            "object": "list",
            "data": [
                {
                    "id": self.served_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ollama",
                    "permission": [],
                    "root": self.served_model_name,
                    "parent": None
                }
            ]
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(models_data).encode())
    
    def handle_health(self):
        try:
            # Check if Ollama is responding
            response = requests.get(f"http://localhost:{self.ollama_port}/api/tags", timeout=5)
            if response.status_code == 200:
                status = {"status": "healthy", "model": self.model_name}
            else:
                status = {"status": "unhealthy", "error": "Ollama server not responding"}
        except Exception as e:
            status = {"status": "unhealthy", "error": str(e)}
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())
    
    def handle_chat_completions(self):
        if not self.check_auth():
            self.send_error(401, 'Unauthorized')
            return
        
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Log minimal request info
            print(f"Chat completion request: model={request_data.get('model', 'unknown')}, messages={len(request_data.get('messages', []))}, stream={request_data.get('stream', False)}")
            
            # Convert OpenAI format to Ollama format
            ollama_request = {
                "model": self.model_name,
                "messages": request_data.get("messages", []),
                "stream": request_data.get("stream", False)
            }
            
            # Map OpenAI parameters to Ollama options
            options = {}
            if "temperature" in request_data:
                options["temperature"] = request_data["temperature"]
            if "max_tokens" in request_data:
                options["num_predict"] = request_data["max_tokens"]
            if "top_p" in request_data:
                options["top_p"] = request_data["top_p"]
            if "frequency_penalty" in request_data:
                options["frequency_penalty"] = request_data["frequency_penalty"]
            if "presence_penalty" in request_data:
                options["presence_penalty"] = request_data["presence_penalty"]
            
            if options:
                ollama_request["options"] = options
            
            # Make request to Ollama
            ollama_url = f"http://localhost:{self.ollama_port}/api/chat"
            
            if ollama_request["stream"]:
                self.handle_streaming_response(ollama_url, ollama_request, request_data)
            else:
                self.handle_non_streaming_response(ollama_url, ollama_request, request_data)
                
        except Exception as e:
            print(f"Error handling chat completions: {e}")
            self.send_error(500, f"Internal server error: {str(e)}")
    
    def handle_streaming_response(self, ollama_url, ollama_request, original_request):
        try:
            response = requests.post(ollama_url, json=ollama_request, stream=True, timeout=300)
            
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            chat_id = f"chatcmpl-{int(time.time())}"
            
            for line in response.iter_lines():
                if line:
                    try:
                        ollama_data = json.loads(line.decode('utf-8'))
                        
                        # Convert Ollama streaming format to OpenAI format
                        openai_data = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": self.served_model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": None
                                }
                            ]
                        }
                        
                        if "message" in ollama_data and "content" in ollama_data["message"]:
                            openai_data["choices"][0]["delta"]["content"] = ollama_data["message"]["content"]
                        
                        if ollama_data.get("done", False):
                            openai_data["choices"][0]["finish_reason"] = "stop"
                            openai_data["choices"][0]["delta"] = {}
                        
                        self.wfile.write(f"data: {json.dumps(openai_data)}\n\n".encode())
                        self.wfile.flush()
                        
                        if ollama_data.get("done", False):
                            self.wfile.write("data: [DONE]\n\n".encode())
                            break
                            
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"Error in streaming response: {e}")
            error_data = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.served_model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            self.wfile.write(f"data: {json.dumps(error_data)}\n\n".encode())
            self.wfile.write("data: [DONE]\n\n".encode())
    
    def handle_non_streaming_response(self, ollama_url, ollama_request, original_request):
        try:
            response = requests.post(ollama_url, json=ollama_request, timeout=300)
            ollama_data = response.json()
            
            # Convert Ollama format to OpenAI format
            openai_response = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.served_model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": ollama_data.get("message", {}).get("content", "")
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": ollama_data.get("prompt_eval_count", 0),
                    "completion_tokens": ollama_data.get("eval_count", 0),
                    "total_tokens": ollama_data.get("prompt_eval_count", 0) + ollama_data.get("eval_count", 0)
                }
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(openai_response).encode())
            
        except Exception as e:
            print(f"Error in non-streaming response: {e}")
            self.send_error(500, f"Internal server error: {str(e)}")
    
    def handle_completions(self):
        if not self.check_auth():
            self.send_error(401, 'Unauthorized')
            return
        
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Log minimal request info
            print(f"Text completion request: model={request_data.get('model', 'unknown')}, prompt_length={len(request_data.get('prompt', ''))}")
            
            # Convert to chat format for Ollama
            prompt = request_data.get("prompt", "")
            ollama_request = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": request_data.get("stream", False)
            }
            
            # Map parameters
            options = {}
            if "temperature" in request_data:
                options["temperature"] = request_data["temperature"]
            if "max_tokens" in request_data:
                options["num_predict"] = request_data["max_tokens"]
            if "top_p" in request_data:
                options["top_p"] = request_data["top_p"]
            
            if options:
                ollama_request["options"] = options
            
            # Make request to Ollama generate endpoint
            ollama_url = f"http://localhost:{self.ollama_port}/api/generate"
            response = requests.post(ollama_url, json=ollama_request, timeout=300)
            ollama_data = response.json()
            
            # Convert to OpenAI completions format
            openai_response = {
                "id": f"cmpl-{int(time.time())}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": self.served_model_name,
                "choices": [
                    {
                        "text": ollama_data.get("response", ""),
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": ollama_data.get("prompt_eval_count", 0),
                    "completion_tokens": ollama_data.get("eval_count", 0),
                    "total_tokens": ollama_data.get("prompt_eval_count", 0) + ollama_data.get("eval_count", 0)
                }
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(openai_response).encode())
            
        except Exception as e:
            print(f"Error handling completions: {e}")
            self.send_error(500, f"Internal server error: {str(e)}")
    
    def handle_embeddings(self):
        if not self.check_auth():
            self.send_error(401, 'Unauthorized')
            return
        
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Log minimal request info  
            input_text = request_data.get("input", "")
            if isinstance(input_text, list):
                input_len = len(input_text[0]) if input_text else 0
            else:
                input_len = len(input_text)
            print(f"Embeddings request: model={request_data.get('model', 'unknown')}, input_length={input_len}")
            
            # Get input text
            input_text = request_data.get("input", "")
            if isinstance(input_text, list):
                input_text = input_text[0] if input_text else ""
            
            # Make request to Ollama embeddings endpoint
            ollama_request = {
                "model": self.model_name,
                "prompt": input_text
            }
            
            response = requests.post(f"http://localhost:{self.ollama_port}/api/embeddings", 
                                   json=ollama_request, timeout=60)
            ollama_data = response.json()
            
            # Convert to OpenAI format
            openai_response = {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": ollama_data.get("embedding", []),
                        "index": 0
                    }
                ],
                "model": self.served_model_name,
                "usage": {
                    "prompt_tokens": len(input_text.split()),
                    "total_tokens": len(input_text.split())
                }
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(openai_response).encode())
            
        except Exception as e:
            print(f"Error handling embeddings: {e}")
            self.send_error(500, f"Internal server error: {str(e)}")
    
    def handle_moderations(self):
        if not self.check_auth():
            self.send_error(401, 'Unauthorized')
            return
        
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Simple moderation response (always safe for Ollama models)
            openai_response = {
                "id": f"modr-{int(time.time())}",
                "model": "text-moderation-stable",
                "results": [
                    {
                        "flagged": False,
                        "categories": {
                            "sexual": False,
                            "hate": False,
                            "violence": False,
                            "self-harm": False,
                            "sexual/minors": False,
                            "hate/threatening": False,
                            "violence/graphic": False
                        },
                        "category_scores": {
                            "sexual": 0.0,
                            "hate": 0.0,
                            "violence": 0.0,
                            "self-harm": 0.0,
                            "sexual/minors": 0.0,
                            "hate/threatening": 0.0,
                            "violence/graphic": 0.0
                        }
                    }
                ]
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(openai_response).encode())
            
        except Exception as e:
            print(f"Error handling moderations: {e}")
            self.send_error(500, f"Internal server error: {str(e)}")
    
    def handle_images_generations(self):
        self.send_error(501, 'Image generation not supported by Ollama')
    
    def handle_audio_transcriptions(self):
        self.send_error(501, 'Audio transcription not supported by Ollama')
    
    def handle_audio_translations(self):
        self.send_error(501, 'Audio translation not supported by Ollama')
    
    # Files API endpoints
    def handle_files_get(self):
        if not self.check_auth():
            self.send_error(401, 'Unauthorized')
            return
        
        # Return empty file list
        response = {
            "object": "list",
            "data": []
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def handle_files_post(self):
        self.send_error(501, 'File upload not supported')
    
    def handle_files_delete(self):
        self.send_error(501, 'File deletion not supported')
    
    # Fine-tuning API endpoints
    def handle_fine_tuning_get(self):
        if not self.check_auth():
            self.send_error(401, 'Unauthorized')
            return
        
        response = {
            "object": "list",
            "data": []
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def handle_fine_tuning_post(self):
        self.send_error(501, 'Fine-tuning not supported by Ollama')
    
    def handle_fine_tuning_delete(self):
        self.send_error(501, 'Fine-tuning not supported by Ollama')
    
    # Assistants API endpoints
    def handle_assistants_get(self):
        if not self.check_auth():
            self.send_error(401, 'Unauthorized')
            return
        
        response = {
            "object": "list",
            "data": []
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def handle_assistants_post(self):
        self.send_error(501, 'Assistants API not supported by Ollama')
    
    def handle_assistants_delete(self):
        self.send_error(501, 'Assistants API not supported by Ollama')
    
    def handle_assistants_patch(self):
        self.send_error(501, 'Assistants API not supported by Ollama')
    
    # Threads API endpoints
    def handle_threads_get(self):
        if not self.check_auth():
            self.send_error(401, 'Unauthorized')
            return
        
        response = {
            "object": "list",
            "data": []
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def handle_threads_post(self):
        self.send_error(501, 'Threads API not supported by Ollama')
    
    def handle_threads_delete(self):
        self.send_error(501, 'Threads API not supported by Ollama')
    
    def handle_threads_patch(self):
        self.send_error(501, 'Threads API not supported by Ollama')
    
    # Messages API endpoints
    def handle_messages_get(self):
        if not self.check_auth():
            self.send_error(401, 'Unauthorized')
            return
        
        response = {
            "object": "list",
            "data": []
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def handle_messages_post(self):
        self.send_error(501, 'Messages API not supported by Ollama')
    
    # Runs API endpoints
    def handle_runs_get(self):
        if not self.check_auth():
            self.send_error(401, 'Unauthorized')
            return
        
        response = {
            "object": "list",
            "data": []
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def handle_runs_post(self):
        self.send_error(501, 'Runs API not supported by Ollama')

def run_proxy():
    port = int(os.environ.get('PORT', 9000))
    server = HTTPServer(('0.0.0.0', port), OllamaProxyHandler)
    print(f"🚀 API server listening on port {port}")
    server.serve_forever()

if __name__ == '__main__':
    run_proxy()
EOF

  chmod +x /tmp/api_proxy.py
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

# Create and start the OpenAI-compatible API proxy
create_api_proxy

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

# Start the API proxy (this will run in foreground)
python3 /tmp/api_proxy.py 