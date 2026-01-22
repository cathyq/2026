# Databricks Agent App

AI Agent with Long-Term Memory using Databricks Lakebase and LangGraph.

## Features

- **Short-term Memory**: Maintains conversation context within a session using PostgresSaver checkpointing
- **Long-term Memory**: Remembers important information across conversations using PostgresStore
- **Multiple Backend Options**: Streamlit (default), FastAPI + Gradio, or MLflow Agent Server
- **MLflow Agent Server**: OpenAI-compatible `/invocations` endpoint with ResponsesAgent schema
- **Automatic Token Refresh**: OAuth tokens are refreshed automatically before expiration
- **Connection Pooling**: Efficient database connection management
- **MLflow Tracing**: Automatic tracing integration for observability

## Architecture

### Streamlit Version (Default)
```
┌─────────────────────────────────────────────────────────────────┐
│                     Databricks Apps                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Streamlit App                          │   │
│  │  ┌─────────────┐              ┌─────────────────────┐   │   │
│  │  │   Chat UI   │─────────────▶│    Memory Agent     │   │   │
│  │  └─────────────┘              └─────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│                     ┌─────────────────────────────────┐        │
│                     │      LangGraph + LangChain      │        │
│                     └─────────────────────────────────┘        │
│                              │                                  │
├──────────────────────────────┼──────────────────────────────────┤
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Databricks Lakebase                     │   │
│  │  ┌─────────────────┐    ┌─────────────────────────────┐ │   │
│  │  │ PostgresSaver   │    │      PostgresStore          │ │   │
│  │  │ (Short-term)    │    │      (Long-term)            │ │   │
│  │  └─────────────────┘    └─────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### FastAPI Version (Alternative)
```
┌─────────────────────────────────────────────────────────────────┐
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │  Gradio UI  │───▶│   FastAPI API   │───▶│  Memory Agent   │ │
│  └─────────────┘    └─────────────────┘    └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### MLflow Agent Server Version (Production)
```
┌─────────────────────────────────────────────────────────────────┐
│                     MLflow Agent Server                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              /invocations Endpoint                       │   │
│  │  ┌─────────────────┐    ┌─────────────────────────────┐ │   │
│  │  │ ResponsesAgent  │───▶│  MemoryAgent (LangGraph)    │ │   │
│  │  │ Request/Response│    │  + Lakebase Persistence     │ │   │
│  │  └─────────────────┘    └─────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│                     ┌─────────────────────────────────┐        │
│                     │        MLflow Tracing           │        │
│                     └─────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
databricks-agent-app/
├── app.yaml                 # Streamlit app config (default)
├── app.fastapi.yaml         # FastAPI app config (alternative)
├── app.agent-server.yaml    # MLflow Agent Server config
├── databricks.yml           # Asset Bundles deployment config
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variables template
├── README.md
├── streamlit_app.py         # Standalone Streamlit application
├── streamlit_local.py       # Local dev Streamlit (in-memory storage)
├── agent_server.py          # MLflow Agent Server handlers
├── start_server.py          # MLflow Agent Server entry point
├── src/
│   ├── __init__.py
│   ├── app.py               # FastAPI application
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # Configuration settings
│   │   └── database.py      # Lakebase connection manager
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── memory_agent.py  # LangGraph agent with memory
│   │   └── responses_agent.py # MLflow ResponsesAgent wrapper
│   ├── models/
│   │   ├── __init__.py
│   │   └── chat.py          # Pydantic models
│   └── routers/
│       ├── __init__.py
│       └── v1/
│           ├── __init__.py
│           ├── chat.py      # Chat API endpoints
│           └── health.py    # Health check endpoints
└── ui/
    ├── chat_ui.py           # Gradio chat interface
    └── streamlit_app.py     # Streamlit chat interface (API client)
```

## Prerequisites

1. **Databricks Workspace** with Apps enabled
2. **Lakebase Instance** created and configured
3. **Databricks CLI** installed and configured
4. **Model Serving Endpoint** (e.g., `databricks-claude-3-5-sonnet`)

## Quick Start

### Option 1: Streamlit (Recommended)

```bash
cd databricks-agent-app
cp .env.example .env
# Edit .env with your configuration

pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

### Option 2: FastAPI + API

```bash
cd databricks-agent-app
cp .env.example .env
# Edit .env with your configuration

pip install -r requirements.txt

# Run FastAPI backend
uvicorn src.app:app --reload --port 8000
```

Open http://localhost:8000/docs for API documentation.

### Option 3: MLflow Agent Server (Production)

The MLflow Agent Server provides an OpenAI-compatible `/invocations` endpoint with automatic request/response validation and MLflow tracing.

```bash
cd databricks-agent-app
cp .env.example .env
# Edit .env with your configuration

pip install -r requirements.txt

# Run MLflow Agent Server
python start_server.py

# Or with uvicorn for development
uvicorn start_server:app --reload --port 8000
```

The server exposes:
- `POST /invocations` - Chat endpoint (ResponsesAgent schema)
- `GET /health` - Health check
- `GET /version` - Version information

## Setup

### 1. Create Lakebase Resources

```sql
-- Create a Lakebase instance (via Databricks UI or API)
-- Instance name: my-agent-lakebase

-- Create database
CREATE DATABASE agent_memory;
```

### 2. Configure Secrets

```bash
# Create secret scope
databricks secrets create-scope agent-app-secrets

# Add secrets
databricks secrets put-secret agent-app-secrets lakebase-instance-name --string-value "my-agent-lakebase"
databricks secrets put-secret agent-app-secrets lakebase-database-name --string-value "agent_memory"
databricks secrets put-secret agent-app-secrets lakebase-catalog-name --string-value "main"
databricks secrets put-secret agent-app-secrets model-endpoint --string-value "databricks-claude-3-5-sonnet"
```

### 3. Deploy to Databricks Apps

**Deploy Streamlit version (default):**
```bash
databricks bundle deploy --target dev
```

**Deploy FastAPI version:**
```bash
# Swap the app.yaml files
mv app.yaml app.streamlit.yaml
mv app.fastapi.yaml app.yaml

databricks bundle deploy --target dev
```

**Deploy MLflow Agent Server version:**
```bash
# Swap the app.yaml files
mv app.yaml app.streamlit.yaml
mv app.agent-server.yaml app.yaml

databricks bundle deploy --target dev
```

## Backend Comparison

| Feature | Streamlit | FastAPI + Gradio | MLflow Agent Server |
|---------|-----------|------------------|---------------------|
| **Simplicity** | Single file | Separate frontend/backend | Single endpoint |
| **API Access** | No REST API | Custom REST API | OpenAI-compatible API |
| **Streaming** | Built-in | SSE endpoint | SSE with ResponsesAgent |
| **Tracing** | Manual | Manual | Automatic MLflow tracing |
| **Validation** | Manual | Pydantic | Automatic ResponsesAgent |
| **Best For** | Quick demos | Custom integrations | Production deployments |

## API Endpoints (FastAPI Version)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/chat` | Send a message and get a response |
| `POST` | `/api/v1/chat/stream` | Stream the response (SSE) |
| `GET` | `/api/v1/chat/history/{thread_id}` | Get conversation history |
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/health/database` | Database connectivity check |

### Example Request

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "My name is Alex and I work at Acme Corp",
    "thread_id": "thread_123",
    "user_id": "user_456"
  }'
```

### Example Response

```json
{
  "message": "Nice to meet you, Alex! I'll remember that you work at Acme Corp. How can I help you today?",
  "thread_id": "thread_123",
  "user_id": "user_456"
}
```

## API Endpoints (MLflow Agent Server)

The MLflow Agent Server exposes a single `/invocations` endpoint that follows the [ResponsesAgent schema](https://mlflow.org/docs/latest/genai/serving/responses-agent/).

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/invocations` | Chat endpoint (set `stream: true` for streaming) |
| `GET` | `/health` | Health check |
| `GET` | `/version` | Version information |

### Example Request (Non-streaming)

```bash
curl -X POST "http://localhost:8000/invocations" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      {"role": "user", "content": "My name is Alex and I work at Acme Corp"}
    ],
    "context": {
      "conversation_id": "thread_123",
      "user_id": "user_456"
    }
  }'
```

### Example Response

```json
{
  "output": [
    {
      "type": "message",
      "id": "msg_abc123",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "Nice to meet you, Alex! I'll remember that you work at Acme Corp."
        }
      ]
    }
  ]
}
```

### Example Request (Streaming)

```bash
curl -X POST "http://localhost:8000/invocations" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      {"role": "user", "content": "What do you know about me?"}
    ],
    "context": {
      "conversation_id": "thread_123",
      "user_id": "user_456"
    },
    "stream": true
  }'
```

The streaming response uses Server-Sent Events (SSE) format with `ResponsesAgentStreamEvent` objects.

## Memory System

### Short-term Memory (PostgresSaver)

- Persists conversation state within a thread
- Enables conversation to be resumed at any point
- Automatically checkpoints at each graph super-step

### Long-term Memory (PostgresStore)

- Stores important information across threads
- Organized by user-specific namespaces
- Enables personalized experiences over time

### Memory Triggers

The agent automatically stores information when users:
- Share their name ("My name is...")
- Express preferences ("I prefer...", "I like...")
- Provide personal info ("I work at...", "I live in...")
- Explicitly request ("Remember that...", "Don't forget...")

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `LAKEBASE_INSTANCE_NAME` | Lakebase instance name | Required |
| `LAKEBASE_DATABASE_NAME` | Database name | Required |
| `LAKEBASE_CATALOG_NAME` | Catalog name | Required |
| `DATABRICKS_MODEL_ENDPOINT` | LLM endpoint | `databricks-claude-3-5-sonnet` |
| `DB_POOL_SIZE` | Connection pool size | `5` |
| `DB_MAX_OVERFLOW` | Max overflow connections | `10` |

## Switching Between Backend Versions

### Switch to FastAPI:
```bash
mv app.yaml app.streamlit.yaml
mv app.fastapi.yaml app.yaml
databricks bundle deploy --target dev
```

### Switch to MLflow Agent Server:
```bash
mv app.yaml app.streamlit.yaml
mv app.agent-server.yaml app.yaml
databricks bundle deploy --target dev
```

### Switch back to Streamlit:
```bash
mv app.yaml app.agent-server.yaml  # or app.fastapi.yaml
mv app.streamlit.yaml app.yaml
databricks bundle deploy --target dev
```

## License

Apache 2.0
