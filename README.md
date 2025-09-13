# Krisp Hackathon 2025 - AI Meeting Assistant Chat

A production-grade AI-powered chat application that interacts with a large proprietary knowledge base of meeting transcriptions, built for the Krisp Hackathon 2025.

## 🚀 Features

- **Advanced RAG System**: Multi-stage retrieval with semantic search and reranking
- **Hybrid Search**: Combines vector similarity with keyword matching for optimal results
- **Context Management**: Smart chunking and context window optimization
- **Streaming Responses**: Real-time chat responses with low latency
- **Audio-Ready Architecture**: Extensible design to incorporate audio files
- **Production Optimizations**: Caching, connection pooling, and cost-aware processing
- **Modern UI**: Streamlit-based chat interface with real-time updates

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit     │────▶│   FastAPI       │────▶│  Vector DB      │
│   Frontend      │     │   Backend       │     │  (Qdrant)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                          │
                               ▼                          ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   LLM Service   │     │  Embeddings     │
                        │   (OpenAI)      │     │  Service        │
                        └─────────────────┘     └─────────────────┘
```

## 📋 Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API Key

## 🚀 Quick Start

### Option 1: Using Docker (Recommended)

1. **Clone and setup**:
```bash
git clone <repo>
cd krisp-hackathon
```

2. **Run the quick start script**:
```bash
./start.sh
```

This will:
- Check for required dependencies
- Create .env file if needed
- Start all services with Docker Compose
- Verify services are running

3. **Access the application**:
- Streamlit App: http://localhost:8501
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Option 2: Manual Setup

1. **Environment setup**:
```bash
cp env.example .env
# Edit .env with your OpenAI API key
```

2. **Start services individually**:
```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Start Redis
docker run -p 6379:6379 redis:7-alpine

# Start Backend (in a new terminal)
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload

# Start Frontend (in a new terminal)
pip install -r requirements.txt
streamlit run app.py
```

## 🔧 Development Setup

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

### Frontend (Streamlit)
```bash
streamlit run app.py
```

## 📊 Data Ingestion

Place your meeting transcription data in `data/transcriptions/` and run:

```bash
python scripts/ingest_data.py --source data/transcriptions --type transcript
```

For audio files (when available):
```bash
python scripts/ingest_data.py --source data/audio --type audio
```

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest

# Frontend tests
pytest tests/test_frontend.py
```

## 📈 Performance & Scaling

- **Caching**: Redis-based caching for embeddings and LLM responses
- **Batch Processing**: Efficient batch embedding generation
- **Connection Pooling**: Optimized database connections
- **Async Operations**: Non-blocking I/O throughout the stack
- **Cost Optimization**: Smart context pruning and response caching

## 🛠️ Configuration

Key configuration options in `.env`:

```
# LLM Settings
OPENAI_API_KEY=your-key
MODEL_NAME=gpt-4-turbo-preview
MAX_CONTEXT_LENGTH=128000

# Vector DB
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Performance
CACHE_TTL=3600
BATCH_SIZE=100
MAX_WORKERS=4
```

## 📚 API Documentation

The backend provides a RESTful API with WebSocket support for real-time chat:

- `POST /api/chat` - Send a chat message
- `WS /api/ws/chat` - WebSocket for streaming chat
- `POST /api/ingest` - Ingest new transcription data
- `GET /api/search` - Search the knowledge base
- `GET /api/health` - Health check endpoint

## 🔐 Security

- API key authentication
- Rate limiting
- Input validation and sanitization
- Secure WebSocket connections

## 📦 Deployment

Production deployment guide available in `docs/deployment.md`

## 👥 Team

Built with ❤️ for Krisp Hackathon 2025
