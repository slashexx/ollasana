#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import signal
import requests
import shutil
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Global variables for orchestration
ollama_process = None
model_name = ""
served_model_name = ""
ollama_port = ""

class OllamaOrchestrator:
    def __init__(self):
        global model_name, served_model_name, ollama_port
        
        model_name = os.environ.get('MODEL_NAME', '').strip()
        served_model_name = os.environ.get('SERVED_MODEL_NAME', model_name)
        ollama_port = os.environ.get('OLLAMA_PORT', '11434')
        
        # Validate required environment variables
        if not model_name:
            print("❌ ERROR: MODEL_NAME environment variable is required")
            print("Please provide an Ollama model name (e.g., llama2, mistral, codellama, etc.)")
            print("See https://ollama.ai/library for available models")
            sys.exit(1)
        
        if not served_model_name:
            served_model_name = model_name
            print(f"SERVED_MODEL_NAME not provided, using: {served_model_name}")
        
        # Set up environment
        self.setup_environment()
        
    def setup_environment(self):
        """Set up necessary directories and environment variables"""
        ollama_models = os.environ.get('OLLAMA_MODELS', '/data-models')
        ollama_home = os.environ.get('OLLAMA_HOME', '/tmp/ollama_home')
        
        Path(ollama_models).mkdir(parents=True, exist_ok=True)
        Path(ollama_home).mkdir(parents=True, exist_ok=True)
        
        print("=== Ollama Model Server Configuration ===")
        print(f"Model: {model_name}")
        print(f"Served Model Name: {served_model_name}")
        print(f"Internal Ollama Port: {ollama_port} (not exposed)")
        print(f"Models Directory: {ollama_models}")
        print("==========================================")
        
    def detect_gpu(self):
        """Detect and configure GPU support"""
        print("=== GPU Detection ===")
        
        # Check for NVIDIA GPU
        if shutil.which('nvidia-smi'):
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print("NVIDIA GPU detected:")
                    print(result.stdout.strip())
                    
                    # Set GPU-specific Ollama environment variables
                    os.environ['OLLAMA_GPU_LAYERS'] = '999'  # Use all GPU layers
                    os.environ['OLLAMA_GPU'] = '1'
                    print("✓ Ollama configured for GPU acceleration")
                else:
                    print("⚠ Could not query GPU details")
                    self._set_cpu_mode()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                print("⚠ nvidia-smi available but failed to execute")
                self._set_cpu_mode()
        else:
            print("⚠ No NVIDIA GPU detected or nvidia-smi not available")
            print("  Running in CPU-only mode")
            self._set_cpu_mode()
        
        # Check for CUDA availability
        if Path("/usr/local/cuda").exists() or os.environ.get('CUDA_HOME'):
            print("✓ CUDA installation detected")
        else:
            print("⚠ CUDA not found - GPU acceleration may not work")
        
        print("========================")
    
    def _set_cpu_mode(self):
        """Configure for CPU-only mode"""
        os.environ['OLLAMA_GPU'] = '0'
        os.environ['OLLAMA_GPU_LAYERS'] = '0'
    
    def start_ollama_server(self):
        """Start Ollama server in background"""
        global ollama_process
        
        print(f"Starting internal Ollama server on localhost:{ollama_port}...")
        
        os.environ['OLLAMA_HOST'] = f'127.0.0.1:{ollama_port}'
        
        try:
            ollama_process = subprocess.Popen(
                ['ollama', 'serve'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for Ollama server to be ready
            print("Waiting for Ollama server to be ready...")
            for i in range(1, 61):  # 60 second timeout
                try:
                    response = requests.get(f"http://localhost:{ollama_port}/api/tags", timeout=2)
                    if response.status_code == 200:
                        print("✓ Ollama server is ready")
                        return
                except requests.exceptions.RequestException:
                    pass
                
                if i == 60:
                    print("✗ Ollama server failed to start within 60 seconds")
                    self.cleanup()
                    sys.exit(1)
                
                print(f"  Waiting... ({i}/60)")
                time.sleep(1)
                
        except FileNotFoundError:
            print("✗ Ollama binary not found. Please ensure Ollama is installed.")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Failed to start Ollama server: {e}")
            sys.exit(1)
    
    def load_ollama_model(self):
        """Pull and load the specified Ollama model"""
        print(f"Loading Ollama model: {model_name}")
        
        # Check if model is already available locally
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and model_name in result.stdout:
                print(f"✓ Model {model_name} is already available locally")
                return
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        print(f"Pulling model {model_name} from Ollama registry...")
        print("This may take several minutes depending on model size...")
        
        try:
            result = subprocess.run(['ollama', 'pull', model_name], 
                                  capture_output=False, timeout=1800)  # 30 min timeout
            if result.returncode == 0:
                print(f"✓ Successfully pulled model: {model_name}")
            else:
                print(f"✗ Failed to pull model: {model_name}")
                print("Please check if the model name is correct.")
                print("Available models can be found at: https://ollama.ai/library")
                self.cleanup()
                sys.exit(1)
        except subprocess.TimeoutExpired:
            print("✗ Model pull timed out after 30 minutes")
            self.cleanup()
            sys.exit(1)
        except Exception as e:
            print(f"✗ Failed to pull model: {e}")
            self.cleanup()
            sys.exit(1)
    
    def validate_model(self):
        """Validate that the model is working correctly"""
        print("Validating model functionality...")
        
        try:
            response = requests.post(
                f"http://localhost:{ollama_port}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Hello",
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'response' in data:
                    print("✓ Model validation successful")
                    return
            
            print("✗ Model validation failed")
            print(f"Response: {response.text}")
            self.cleanup()
            sys.exit(1)
            
        except Exception as e:
            print(f"✗ Model validation failed: {e}")
            self.cleanup()
            sys.exit(1)
    
    def cleanup(self):
        """Clean up processes on exit"""
        global ollama_process
        print("Shutting down services...")
        if ollama_process and ollama_process.poll() is None:
            try:
                ollama_process.terminate()
                ollama_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ollama_process.kill()
            except Exception:
                pass
    
    def run_setup(self):
        """Run the complete setup process"""
        print("🦙 Starting Ollama-based OpenAI-compatible API server")
        
        # Detect and configure GPU
        self.detect_gpu()
        
        # Start Ollama server
        self.start_ollama_server()
        
        # Load the Ollama model
        self.load_ollama_model()
        
        # Validate model is working
        self.validate_model()
        
        print("✅ Ollama setup completed successfully!")

# Initialize orchestrator
orchestrator = OllamaOrchestrator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting application...")
    orchestrator.run_setup()
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        orchestrator.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    yield
    
    # Shutdown
    print("🔄 Shutting down application...")
    orchestrator.cleanup()

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Ollama OpenAI-Compatible API",
    description="OpenAI-compatible API server powered by Ollama",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
API_KEY = os.environ.get('API_KEY', '')

# Security dependency
security = HTTPBearer(auto_error=False)

def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not API_KEY:
        return True
    
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stream: Optional[bool] = False

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False

class EmbeddingRequest(BaseModel):
    model: str
    input: str

class ModerationRequest(BaseModel):
    input: str
    model: Optional[str] = "text-moderation-stable"

# API endpoints
@app.get("/")
async def root():
    return {"details": "not found"}

@app.get("/health")
@app.get("/v1/health")
async def health_check():
    try:
        response = requests.get(f"http://localhost:{ollama_port}/api/tags", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "model": model_name}
        else:
            return {"status": "unhealthy", "error": "Ollama server not responding"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/v1/models")
async def list_models(auth: bool = Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [{
            "id": served_model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "ollama",
            "permission": [],
            "root": served_model_name,
            "parent": None
        }]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, auth: bool = Depends(verify_api_key)):
    print(f"Chat completion request: model={request.model}, messages={len(request.messages)}, stream={request.stream}")
    
    ollama_request = {
        "model": model_name,
        "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
        "stream": request.stream
    }
    
    options = {}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.max_tokens is not None:
        options["num_predict"] = request.max_tokens
    if request.top_p is not None:
        options["top_p"] = request.top_p
    if request.frequency_penalty is not None:
        options["frequency_penalty"] = request.frequency_penalty
    if request.presence_penalty is not None:
        options["presence_penalty"] = request.presence_penalty
    
    if options:
        ollama_request["options"] = options
    
    ollama_url = f"http://localhost:{ollama_port}/api/chat"
    
    try:
        if request.stream:
            return StreamingResponse(
                stream_chat_response(ollama_url, ollama_request),
                media_type="text/event-stream"
            )
        else:
            response = requests.post(ollama_url, json=ollama_request, timeout=300)
            ollama_data = response.json()
            
            openai_response = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": served_model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": ollama_data.get("message", {}).get("content", "")
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": ollama_data.get("prompt_eval_count", 0),
                    "completion_tokens": ollama_data.get("eval_count", 0),
                    "total_tokens": ollama_data.get("prompt_eval_count", 0) + ollama_data.get("eval_count", 0)
                }
            }
            
            return openai_response
            
    except Exception as e:
        print(f"Error handling chat completions: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def stream_chat_response(ollama_url: str, ollama_request: dict):
    try:
        response = requests.post(ollama_url, json=ollama_request, stream=True, timeout=300)
        chat_id = f"chatcmpl-{int(time.time())}"
        
        for line in response.iter_lines():
            if line:
                try:
                    ollama_data = json.loads(line.decode('utf-8'))
                    
                    openai_data = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": served_model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": None
                        }]
                    }
                    
                    if "message" in ollama_data and "content" in ollama_data["message"]:
                        openai_data["choices"][0]["delta"]["content"] = ollama_data["message"]["content"]
                    
                    if ollama_data.get("done", False):
                        openai_data["choices"][0]["finish_reason"] = "stop"
                        openai_data["choices"][0]["delta"] = {}
                    
                    yield f"data: {json.dumps(openai_data)}\n\n"
                    
                    if ollama_data.get("done", False):
                        yield "data: [DONE]\n\n"
                        break
                        
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        print(f"Error in streaming response: {e}")
        error_data = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": served_model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"

# Additional endpoints (shortened for brevity)
@app.post("/v1/completions")
async def completions(request: CompletionRequest, auth: bool = Depends(verify_api_key)):
    # Implementation similar to chat_completions but for text completion
    pass

@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest, auth: bool = Depends(verify_api_key)):
    # Implementation for embeddings
    pass

@app.post("/v1/moderations")
async def moderations(request: ModerationRequest, auth: bool = Depends(verify_api_key)):
    return {
        "id": f"modr-{int(time.time())}",
        "model": "text-moderation-stable",
        "results": [{
            "flagged": False,
            "categories": {
                "sexual": False, "hate": False, "violence": False,
                "self-harm": False, "sexual/minors": False,
                "hate/threatening": False, "violence/graphic": False
            },
            "category_scores": {
                "sexual": 0.0, "hate": 0.0, "violence": 0.0,
                "self-harm": 0.0, "sexual/minors": 0.0,
                "hate/threatening": 0.0, "violence/graphic": 0.0
            }
        }]
    }

# Placeholder endpoints
@app.get("/v1/{endpoint:path}")
async def placeholder_get(auth: bool = Depends(verify_api_key)):
    return {"object": "list", "data": []}

@app.post("/v1/{endpoint:path}")
async def placeholder_post():
    raise HTTPException(status_code=501, detail="Not implemented")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get('PORT', 9000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info") 