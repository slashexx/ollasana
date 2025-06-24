#!/usr/bin/env python3
import json
import requests
import os
import time
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Ollama OpenAI-Compatible API",
    description="OpenAI-compatible API server powered by Ollama",
    version="1.0.0"
)

# Add CORS middleware - this is the proper way to handle CORS in FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configuration from environment variables
API_KEY = os.environ.get('API_KEY', '')
SERVED_MODEL_NAME = os.environ.get('SERVED_MODEL_NAME', '')
MODEL_NAME = os.environ.get('MODEL_NAME', '')
OLLAMA_PORT = os.environ.get('OLLAMA_PORT', '11434')

# Security dependency
security = HTTPBearer(auto_error=False)

def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not API_KEY:
        return True  # No API key required
    
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True

# Pydantic models for request/response validation
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

# Root endpoint
@app.get("/")
async def root():
    return {"details": "not found"}

# Health check endpoint
@app.get("/health")
@app.get("/v1/health")
async def health_check():
    try:
        response = requests.get(f"http://localhost:{OLLAMA_PORT}/api/tags", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "model": MODEL_NAME}
        else:
            return {"status": "unhealthy", "error": "Ollama server not responding"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# Models endpoint
@app.get("/v1/models")
async def list_models(auth: bool = Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollama",
                "permission": [],
                "root": SERVED_MODEL_NAME,
                "parent": None
            }
        ]
    }

# Chat completions endpoint
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, auth: bool = Depends(verify_api_key)):
    print(f"Chat completion request: model={request.model}, messages={len(request.messages)}, stream={request.stream}")
    
    # Convert OpenAI format to Ollama format
    ollama_request = {
        "model": MODEL_NAME,
        "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
        "stream": request.stream
    }
    
    # Map OpenAI parameters to Ollama options
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
    
    # Make request to Ollama
    ollama_url = f"http://localhost:{OLLAMA_PORT}/api/chat"
    
    try:
        if request.stream:
            return StreamingResponse(
                stream_chat_response(ollama_url, ollama_request),
                media_type="text/event-stream"
            )
        else:
            response = requests.post(ollama_url, json=ollama_request, timeout=300)
            ollama_data = response.json()
            
            # Convert Ollama format to OpenAI format
            openai_response = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": SERVED_MODEL_NAME,
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
                    
                    # Convert Ollama streaming format to OpenAI format
                    openai_data = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": SERVED_MODEL_NAME,
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
            "model": SERVED_MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"

# Text completions endpoint
@app.post("/v1/completions")
async def completions(request: CompletionRequest, auth: bool = Depends(verify_api_key)):
    print(f"Text completion request: model={request.model}, prompt_length={len(request.prompt)}")
    
    # Convert to Ollama format
    ollama_request = {
        "model": MODEL_NAME,
        "prompt": request.prompt,
        "stream": request.stream
    }
    
    # Map parameters
    options = {}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.max_tokens is not None:
        options["num_predict"] = request.max_tokens
    if request.top_p is not None:
        options["top_p"] = request.top_p
    
    if options:
        ollama_request["options"] = options
    
    try:
        # Make request to Ollama generate endpoint
        ollama_url = f"http://localhost:{OLLAMA_PORT}/api/generate"
        response = requests.post(ollama_url, json=ollama_request, timeout=300)
        ollama_data = response.json()
        
        # Convert to OpenAI completions format
        openai_response = {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": SERVED_MODEL_NAME,
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
        
        return openai_response
        
    except Exception as e:
        print(f"Error handling completions: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Embeddings endpoint
@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest, auth: bool = Depends(verify_api_key)):
    input_text = request.input
    if isinstance(input_text, list):
        input_text = input_text[0] if input_text else ""
    
    print(f"Embeddings request: model={request.model}, input_length={len(input_text)}")
    
    try:
        # Make request to Ollama embeddings endpoint
        ollama_request = {
            "model": MODEL_NAME,
            "prompt": input_text
        }
        
        response = requests.post(f"http://localhost:{OLLAMA_PORT}/api/embeddings", 
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
            "model": SERVED_MODEL_NAME,
            "usage": {
                "prompt_tokens": len(input_text.split()),
                "total_tokens": len(input_text.split())
            }
        }
        
        return openai_response
        
    except Exception as e:
        print(f"Error handling embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Moderations endpoint
@app.post("/v1/moderations")
async def moderations(request: ModerationRequest, auth: bool = Depends(verify_api_key)):
    # Simple moderation response (always safe for Ollama models)
    return {
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

# Placeholder endpoints for compatibility
@app.get("/v1/files")
@app.get("/v1/fine-tuning/jobs")
@app.get("/v1/assistants")
@app.get("/v1/threads")
@app.get("/v1/messages")
@app.get("/v1/runs")
async def placeholder_get(auth: bool = Depends(verify_api_key)):
    return {"object": "list", "data": []}

@app.post("/v1/images/generations")
@app.post("/v1/audio/transcriptions")
@app.post("/v1/audio/translations")
@app.post("/v1/files")
@app.post("/v1/fine-tuning/jobs")
@app.post("/v1/assistants")
@app.post("/v1/threads")
@app.post("/v1/messages")
@app.post("/v1/runs")
async def placeholder_post():
    raise HTTPException(status_code=501, detail="Not implemented")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get('PORT', 9000))
    print(f"🚀 FastAPI server starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info") 