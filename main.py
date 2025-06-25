#!/usr/bin/env python3
import os
import sys

# Force unbuffered output for real-time logs in containers
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import time
import subprocess
import signal
import requests
import shutil
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

# Global configuration
MODEL_NAME = os.environ.get('MODEL_NAME', '').strip()
SERVED_MODEL_NAME = os.environ.get('SERVED_MODEL_NAME', MODEL_NAME)
PORT = int(os.environ.get('PORT', 9000))
OLLAMA_PORT = int(os.environ.get('OLLAMA_PORT', 11434))
API_KEY = os.environ.get('API_KEY', '')
VALIDATION_TIMEOUT = int(os.environ.get('VALIDATION_TIMEOUT', 0))

# Global state
ollama_process = None

# Validate required environment
if not MODEL_NAME:
    print("❌ ERROR: MODEL_NAME environment variable is required")
    sys.exit(1)

if not SERVED_MODEL_NAME:
    SERVED_MODEL_NAME = MODEL_NAME

print("=== Ollama GPU-Only Server ===", flush=True)
print(f"Model: {MODEL_NAME}", flush=True)
print(f"Port: {PORT}", flush=True)
print("===============================", flush=True)

def detect_gpu():
    """GPU Detection - MANDATORY"""
    print("=== GPU Detection ===", flush=True)
    
    gpu_available = False
    
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("GPU detected:", flush=True)
                print(result.stdout.strip(), flush=True)
                gpu_available = True
            else:
                gpu_available = True  # Try anyway
        except:
            gpu_available = True  # Try anyway
    
    if gpu_available:
        os.environ['OLLAMA_GPU_LAYERS'] = '999'
        os.environ['OLLAMA_GPU'] = '1'
        os.environ['OLLAMA_LLM_LIBRARY'] = 'cuda_v12'
        print(f"🚀 GPU enabled for: {MODEL_NAME}", flush=True)
        print("✓ GPU acceleration configured", flush=True)
    else:
        print("✗ FATAL: No GPU available")
        print("✗ GPU required for deployment")
        sys.exit(1)
    
    print("========================")

def start_ollama():
    """Start Ollama server"""
    global ollama_process
    print(f"Starting Ollama server...", flush=True)
    
    os.environ['OLLAMA_HOST'] = f'127.0.0.1:{OLLAMA_PORT}'
    
    try:
        ollama_process = subprocess.Popen(['ollama', 'serve'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("Waiting for Ollama...", flush=True)
        for i in range(60):
            try:
                response = requests.get(f"http://localhost:{OLLAMA_PORT}/api/tags", timeout=2)
                if response.status_code == 200:
                    print("✓ Ollama ready", flush=True)
                    return
            except:
                pass
            
            if i == 59:
                print("✗ Ollama failed to start")
                sys.exit(1)
            
            time.sleep(1)
            
    except Exception as e:
        print(f"✗ Ollama error: {e}")
        sys.exit(1)

def load_model():
    """Load model"""
    print(f"Loading: {MODEL_NAME}", flush=True)
    
    try:
        result = subprocess.run(['ollama', 'pull', MODEL_NAME], timeout=1800)
        if result.returncode == 0:
            print(f"✓ Model loaded: {MODEL_NAME}", flush=True)
        else:
            print(f"✗ Failed to load: {MODEL_NAME}", flush=True)
            sys.exit(1)
    except Exception as e:
        print(f"✗ Model error: {e}")
        sys.exit(1)

def validate_model():
    """Validate model"""
    print("Validating model...", flush=True)
    
    timeout = VALIDATION_TIMEOUT if VALIDATION_TIMEOUT > 0 else (1800 if any(x in MODEL_NAME.lower() for x in ['34b', '70b', 'llava']) else 600)
    print(f"⏳ Timeout: {timeout}s", flush=True)
    
    try:
        response = requests.post(
            f"http://localhost:{OLLAMA_PORT}/api/generate",
            json={"model": MODEL_NAME, "prompt": "Hi", "stream": False, "options": {"num_predict": 5}},
            timeout=timeout
        )
        
        if response.status_code == 200 and response.json().get('response'):
            print("✓ Validation successful", flush=True)
        else:
            print("✗ Validation failed", flush=True)
            sys.exit(1)
            
    except Exception as e:
        print(f"✗ Validation error: {e}")
        sys.exit(1)

def cleanup():
    """Cleanup"""
    global ollama_process
    if ollama_process:
        try:
            ollama_process.terminate()
            ollama_process.wait(timeout=5)
        except:
            ollama_process.kill() if ollama_process else None

signal.signal(signal.SIGTERM, lambda s, f: (cleanup(), sys.exit(0)))
signal.signal(signal.SIGINT, lambda s, f: (cleanup(), sys.exit(0)))

# FastAPI App
app = FastAPI(title="Ollama OpenAI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

security = HTTPBearer(auto_error=False)

def verify_key(creds = Depends(security)):
    if API_KEY and (not creds or creds.credentials != API_KEY):
        raise HTTPException(401, "Invalid API key")
    return True

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

@app.get("/health")
async def health():
    try:
        r = requests.get(f"http://localhost:{OLLAMA_PORT}/api/tags", timeout=5)
        return {"status": "healthy" if r.status_code == 200 else "unhealthy"}
    except:
        return {"status": "unhealthy"}

@app.get("/v1/models")
async def models(auth = Depends(verify_key)):
    return {"object": "list", "data": [{"id": SERVED_MODEL_NAME, "object": "model", "created": int(time.time())}]}

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest, auth = Depends(verify_key)):
    try:
        req = {"model": MODEL_NAME, "messages": [{"role": m.role, "content": m.content} for m in request.messages]}
        if request.temperature: req["options"] = {"temperature": request.temperature}
        if request.max_tokens: req.setdefault("options", {})["num_predict"] = request.max_tokens
        
        r = requests.post(f"http://localhost:{OLLAMA_PORT}/api/chat", json=req, timeout=300)
        data = r.json()
        
        return {
            "id": f"chat-{int(time.time())}",
            "object": "chat.completion",
            "model": SERVED_MODEL_NAME,
            "choices": [{
                "message": {"role": "assistant", "content": data.get("message", {}).get("content", "")},
                "finish_reason": "stop"
            }]
        }
    except Exception as e:
        raise HTTPException(500, str(e))

def main():
    print("🦙 Starting Ollama GPU Server", flush=True)
    
    try:
        detect_gpu()
        start_ollama()
        load_model()
        validate_model()
        
        print("✅ Ready!", flush=True)
        print(f"🌐 API: http://0.0.0.0:{PORT}", flush=True)
        print("", flush=True)
        
        uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        cleanup()

if __name__ == "__main__":
    main() 