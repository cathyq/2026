# Databricks Agent App

AI Agent with Long-Term Memory using Databricks Lakebase and LangGraph.

## Features

- **Short-term Memory**: Maintains conversation context within a session using PostgresSaver checkpointing
- **Long-term Memory**: Remembers important information across conversations using PostgresStore
- **FastAPI Backend**: Production-ready REST API with streaming support
- **Gradio UI**: Interactive chat interface
- **Automatic Token Refresh**: OAuth tokens are refreshed automatically before expiration
- **Connection Pooling**: Efficient database connection management

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Databricks Apps                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │  Gradio UI  │───▶│   FastAPI API   │───▶│  Memory Agent   │ │
│  └─────────────┘    └─────────────────┘    └─────────────────┘ │
│                              │                      │           │
│                              ▼                      ▼           │
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

## Project Structure

```
databricks-agent-app/
├── app.yaml                 # Databricks Apps configuration
├── databricks.yml           # Asset Bundles deployment config
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
├── README.md
├── src/
│   ├── __init__.py
│   ├── app.py              # FastAPI application
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py       # Configuration settings
│   │   └── database.py     # Lakebase connection manager
│   ├── agent/
│   │   ├── __init__.py
│   │   └── memory_agent.py # LangGraph agent with memory
│   ├── models/
│   │   ├── __init__.py
│   │   └── chat.py         # Pydantic models
│   └── routers/
│       ├── __init__.py
│       └── v1/
│           ├── __init__.py
│           ├── chat.py     # Chat API endpoints
│           └── health.py   # Health check endpoints
└── ui/
    └── chat_ui.py          # Gradio chat interface
```

## Prerequisites

1. **Databricks Workspace** with Apps enabled
2. **Lakebase Instance** created and configured
3. **Databricks CLI** installed and configured
4. **Model Serving Endpoint** (e.g., `databricks-claude-3-5-sonnet`)

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

### 3. Local Development

```bash
# Clone and setup
cd databricks-agent-app
cp .env.example .env
# Edit .env with your configuration

# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn src.app:app --reload --port 8000
```

### 4. Deploy to Databricks Apps

```bash
# Validate bundle
databricks bundle validate

# Deploy to dev
databricks bundle deploy --target dev

# Deploy to production
databricks bundle deploy --target prod
```

## API Endpoints

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

## License

Apache 2.0
