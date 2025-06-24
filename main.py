#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import signal
import requests
import shutil
from pathlib import Path

class OllamaOrchestrator:
    def __init__(self):
        self.model_name = os.environ.get('MODEL_NAME', '').strip()
        self.served_model_name = os.environ.get('SERVED_MODEL_NAME', self.model_name)
        self.port = int(os.environ.get('PORT', 9000))
        self.ollama_port = os.environ.get('OLLAMA_PORT', '11434')
        self.ollama_process = None
        
        # Validate required environment variables
        if not self.model_name:
            print("❌ ERROR: MODEL_NAME environment variable is required")
            print("Please provide an Ollama model name (e.g., llama2, mistral, codellama, etc.)")
            print("See https://ollama.ai/library for available models")
            sys.exit(1)
        
        if not self.served_model_name:
            self.served_model_name = self.model_name
            print(f"SERVED_MODEL_NAME not provided, using: {self.served_model_name}")
        
        # Set up environment
        self.setup_environment()
        
    def setup_environment(self):
        """Set up necessary directories and environment variables"""
        ollama_models = os.environ.get('OLLAMA_MODELS', '/data-models')
        ollama_home = os.environ.get('OLLAMA_HOME', '/tmp/ollama_home')
        
        Path(ollama_models).mkdir(parents=True, exist_ok=True)
        Path(ollama_home).mkdir(parents=True, exist_ok=True)
        
        print("=== Ollama Model Server Configuration ===")
        print(f"Model: {self.model_name}")
        print(f"Served Model Name: {self.served_model_name}")
        print(f"Unified API Port: {self.port}")
        print(f"Internal Ollama Port: {self.ollama_port} (not exposed)")
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
        print(f"Starting internal Ollama server on localhost:{self.ollama_port}...")
        
        os.environ['OLLAMA_HOST'] = f'127.0.0.1:{self.ollama_port}'
        
        try:
            self.ollama_process = subprocess.Popen(
                ['ollama', 'serve'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for Ollama server to be ready
            print("Waiting for Ollama server to be ready...")
            for i in range(1, 61):  # 60 second timeout
                try:
                    response = requests.get(f"http://localhost:{self.ollama_port}/api/tags", timeout=2)
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
        print(f"Loading Ollama model: {self.model_name}")
        
        # Check if model is already available locally
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and self.model_name in result.stdout:
                print(f"✓ Model {self.model_name} is already available locally")
                return
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        print(f"Pulling model {self.model_name} from Ollama registry...")
        print("This may take several minutes depending on model size...")
        
        try:
            result = subprocess.run(['ollama', 'pull', self.model_name], 
                                  capture_output=False, timeout=1800)  # 30 min timeout
            if result.returncode == 0:
                print(f"✓ Successfully pulled model: {self.model_name}")
            else:
                print(f"✗ Failed to pull model: {self.model_name}")
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
                f"http://localhost:{self.ollama_port}/api/generate",
                json={
                    "model": self.model_name,
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
    
    def start_fastapi_server(self):
        """Start the FastAPI server"""
        print("✅ All services started successfully!")
        print("")
        print("🌐 Unified OpenAI-Compatible API Server")
        print(f"Model: {self.model_name}")
        print(f"Public API Endpoint: http://0.0.0.0:{self.port}")
        print(f"Internal Ollama Server: http://localhost:{self.ollama_port} (internal only)")
        print("")
        print("📋 Available endpoints:")
        print("   GET  /v1/models")
        print("   GET  /health")
        print("   POST /v1/chat/completions")
        print("   POST /v1/completions")
        print("   POST /v1/embeddings")
        print("   POST /v1/moderations")
        print("   + Additional OpenAI-compatible endpoints")
        print("")
        print(f"🔗 Access your API at: http://localhost:{self.port}")
        print("")
        print("Starting FastAPI server...")
        
        # Import and run FastAPI server
        import uvicorn
        from api_server import app
        
        uvicorn.run(app, host="0.0.0.0", port=self.port, log_level="info")
    
    def cleanup(self):
        """Clean up processes on exit"""
        print("Shutting down services...")
        if self.ollama_process and self.ollama_process.poll() is None:
            try:
                self.ollama_process.terminate()
                self.ollama_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ollama_process.kill()
            except Exception:
                pass
    
    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def run(self):
        """Main orchestration method"""
        try:
            print("🦙 Starting Ollama-based OpenAI-compatible API server")
            
            # Set up signal handlers
            self.setup_signal_handlers()
            
            # Detect and configure GPU
            self.detect_gpu()
            
            # Start Ollama server
            self.start_ollama_server()
            
            # Load the Ollama model
            self.load_ollama_model()
            
            # Validate model is working
            self.validate_model()
            
            # Start the FastAPI server (runs in foreground)
            self.start_fastapi_server()
            
        except KeyboardInterrupt:
            print("\n⚠ Received interrupt signal")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            self.cleanup()

if __name__ == "__main__":
    orchestrator = OllamaOrchestrator()
    orchestrator.run() 