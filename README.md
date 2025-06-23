# Ollama OpenAI-Compatible API Server

This project provides an OpenAI-compatible API wrapper around Ollama, allowing you to use any Ollama model with OpenAI-compatible clients and tools.

## Features

- 🦙 **Ollama Integration**: Run any Ollama model locally
- 🔌 **OpenAI-Compatible**: Works with existing OpenAI client libraries
- 🚀 **GPU Support**: Automatic NVIDIA GPU detection and acceleration
- 🔒 **Optional Authentication**: API key support for secure access
- 📊 **Health Monitoring**: Built-in health check endpoints
- 🌐 **CORS Enabled**: Ready for web applications

## Quick Start

### Environment Variables

```bash
# Required
MODEL_NAME=llama2              # Ollama model to serve
PORT=9000                      # API server port (default: 9000)
OLLAMA_PORT=11434             # Internal Ollama port (default: 11434)

# Optional
SERVED_MODEL_NAME=my-model     # Model name exposed via API (defaults to MODEL_NAME)
API_KEY=your-secret-key        # Optional API key for authentication
OLLAMA_MODELS=/app/models      # Model storage directory
OLLAMA_HOME=/app/ollama        # Ollama home directory
```

### Docker Usage

```bash
# Pull and run with a specific model
docker run -d \
  --name ollama-api \
  -p 9000:9000 \
  -e MODEL_NAME=llama2 \
  -e API_KEY=your-secret-key \
  your-image-name
```

### Local Usage

```bash
# Set environment variables
export MODEL_NAME=llama2
export PORT=9000

# Run the script
./start-ollama.sh
```

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| `GET` | `/` | Default endpoint | ✅ Working |
| `GET` | `/health` | Health check | ✅ Working |
| `GET` | `/v1/health` | Health check (OpenAI style) | ✅ Working |
| `GET` | `/v1/models` | List available models | ✅ Working |

### Chat & Completions

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| `POST` | `/v1/chat/completions` | Chat completions (streaming & non-streaming) | ✅ Working |
| `POST` | `/v1/completions` | Text completions | ✅ Working |
| `POST` | `/v1/embeddings` | Generate embeddings | ✅ Working |
| `POST` | `/v1/moderations` | Content moderation | ✅ Working |

### Unsupported Endpoints

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| `POST` | `/v1/images/generations` | Image generation | ❌ Not supported |
| `POST` | `/v1/audio/transcriptions` | Audio transcription | ❌ Not supported |
| `POST` | `/v1/audio/translations` | Audio translation | ❌ Not supported |
| `GET/POST/DELETE` | `/v1/files/*` | File management | ❌ Not supported |
| `GET/POST/DELETE` | `/v1/fine-tuning/*` | Fine-tuning | ❌ Not supported |
| `GET/POST/DELETE/PATCH` | `/v1/assistants/*` | Assistants API | ❌ Not supported |
| `GET/POST/DELETE/PATCH` | `/v1/threads/*` | Threads API | ❌ Not supported |
| `GET/POST` | `/v1/messages/*` | Messages API | ❌ Not supported |
| `GET/POST` | `/v1/runs/*` | Runs API | ❌ Not supported |

## Sample Requests

### Authentication

If `API_KEY` is set, include it in the Authorization header:

```bash
-H "Authorization: Bearer your-secret-key"
```

### 1. Default Endpoint

```bash
curl -X GET http://localhost:9000/
```

**Response:**
```json
{
  "details": "not found"
}
```

### 2. Health Check

```bash
curl -X GET http://localhost:9000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "llama2"
}
```

### 3. List Models

```bash
curl -X GET http://localhost:9000/v1/models \
  -H "Authorization: Bearer your-secret-key"
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama2",
      "object": "model",
      "created": 1699000000,
      "owned_by": "ollama",
      "permission": [],
      "root": "llama2",
      "parent": null
    }
  ]
}
```

### 4. Chat Completions

```bash
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "model": "llama2",
    "messages": [
      {
        "role": "user",
        "content": "Hello! How are you?"
      }
    ],
    "temperature": 0.7,
    "max_tokens": 150
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-1699000000",
  "object": "chat.completion",
  "created": 1699000000,
  "model": "llama2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking. How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 17,
    "total_tokens": 29
  }
}
```

### 5. Chat Completions (Streaming)

```bash
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "model": "llama2",
    "messages": [
      {
        "role": "user",
        "content": "Tell me a short story"
      }
    ],
    "stream": true
  }'
```

**Response:** (Server-Sent Events)
```
data: {"id":"chatcmpl-1699000000","object":"chat.completion.chunk","created":1699000000,"model":"llama2","choices":[{"index":0,"delta":{"content":"Once"},"finish_reason":null}]}

data: {"id":"chatcmpl-1699000000","object":"chat.completion.chunk","created":1699000000,"model":"llama2","choices":[{"index":0,"delta":{"content":" upon"},"finish_reason":null}]}

data: [DONE]
```

### 6. Text Completions

```bash
curl -X POST http://localhost:9000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "model": "llama2",
    "prompt": "The capital of France is",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

**Response:**
```json
{
  "id": "cmpl-1699000000",
  "object": "text_completion",
  "created": 1699000000,
  "model": "llama2",
  "choices": [
    {
      "text": " Paris. It is located in the north-central part of the country.",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 14,
    "total_tokens": 20
  }
}
```

### 7. Embeddings

```bash
curl -X POST http://localhost:9000/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "model": "llama2",
    "input": "Hello world"
  }'
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, 0.3, ...],
      "index": 0
    }
  ],
  "model": "llama2",
  "usage": {
    "prompt_tokens": 2,
    "total_tokens": 2
  }
}
```

### 8. Content Moderation

```bash
curl -X POST http://localhost:9000/v1/moderations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "input": "I want to harm someone"
  }'
```

**Response:**
```json
{
  "id": "modr-1699000000",
  "model": "text-moderation-stable",
  "results": [
    {
      "flagged": false,
      "categories": {
        "sexual": false,
        "hate": false,
        "violence": false,
        "self-harm": false,
        "sexual/minors": false,
        "hate/threatening": false,
        "violence/graphic": false
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
```

## Client Library Usage

### Python (OpenAI Library)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9000/v1",
    api_key="your-secret-key"  # Optional
)

# Chat completion
response = client.chat.completions.create(
    model="llama2",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### JavaScript/Node.js

```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
  baseURL: 'http://localhost:9000/v1',
  apiKey: 'your-secret-key' // Optional
});

const response = await openai.chat.completions.create({
  model: 'llama2',
  messages: [
    { role: 'user', content: 'Hello!' }
  ]
});

console.log(response.choices[0].message.content);
```

## Available Ollama Models

Visit [Ollama Library](https://ollama.ai/library) for a complete list of available models:

- `llama2` - Meta's Llama 2 model
- `mistral` - Mistral 7B model
- `codellama` - Code-focused Llama model
- `dolphin-mixtral` - Uncensored Mixtral model
- `neural-chat` - Intel's neural chat model
- And many more...

## Error Responses

### 401 Unauthorized
```json
{
  "error": {
    "message": "Unauthorized",
    "type": "invalid_request_error",
    "code": 401
  }
}
```

### 501 Not Implemented
```json
{
  "error": {
    "message": "Image generation not supported by Ollama",
    "type": "not_implemented_error", 
    "code": 501
  }
}
```

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the model name exists in Ollama library
2. **GPU not detected**: Install NVIDIA drivers and CUDA toolkit
3. **Connection refused**: Check if Ollama service is running
4. **Timeout errors**: Large models may take time to load initially

### Logs

The server provides detailed logging for requests and errors. Monitor the console output for debugging information.

## Contributing

Feel free to submit issues and pull requests to improve this OpenAI-compatible wrapper for Ollama.
Hello world

## License

This project is open source and available under the MIT License. 