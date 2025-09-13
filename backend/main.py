"""Main FastAPI application."""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
from typing import List, Dict, Any, Optional
import uuid
import structlog
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse

from backend.config import settings
from backend.models import (
    ChatRequest, ChatResponse, ChatMessage, MessageRole,
    SearchRequest, SearchResult, IngestRequest, IngestResponse
)
from backend.services.rag import rag_service
from backend.services.llm import llm_service
from backend.services.ingestion import ingestion_service
from backend.services.vector_db import vector_db


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

# Metrics
chat_requests = Counter('chat_requests_total', 'Total number of chat requests')
chat_errors = Counter('chat_errors_total', 'Total number of chat errors')
chat_duration = Histogram('chat_duration_seconds', 'Chat request duration')
websocket_connections = Counter('websocket_connections_total', 'Total WebSocket connections')

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory conversation storage (use Redis in production)
conversations: Dict[str, List[ChatMessage]] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting up application", app_name=settings.app_name)
    
    # Initialize vector database
    await vector_db.initialize()
    
    logger.info("Application startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down application")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "healthy"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check vector DB connection
        stats = await vector_db.get_collection_stats()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "vector_db": {
                "connected": True,
                "vectors_count": stats.get("vectors_count", 0)
            }
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(generate_latest())


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with RAG."""
    chat_requests.inc()
    
    with chat_duration.time():
        try:
            # Get or create conversation
            if request.conversation_id and request.conversation_id in conversations:
                conversation_history = conversations[request.conversation_id]
            else:
                conversation_history = []
                request.conversation_id = request.conversation_id or str(uuid.uuid4())
                conversations[request.conversation_id] = conversation_history
            
            # Add user message to history
            user_message = ChatMessage(
                role=MessageRole.USER,
                content=request.message
            )
            conversation_history.append(user_message)
            
            # Retrieve relevant context using RAG
            chunks, retrieval_metadata = await rag_service.retrieve_context(
                query=request.message,
                conversation_history=conversation_history[:-1],  # Exclude current message
                filters=request.search_filters,
                max_context_tokens=request.context_window
            )
            
            # Format context for LLM
            context = await rag_service.format_context_for_llm(
                chunks=chunks,
                query=request.message
            )
            
            # Generate response
            response_text = await llm_service.generate_response(
                messages=conversation_history,
                context=context
            )
            
            # Add assistant response to history
            assistant_message = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response_text
            )
            conversation_history.append(assistant_message)
            
            # Prepare sources for response
            sources = [
                {
                    "meeting_id": chunk.meeting_id,
                    "chunk_id": chunk.id,
                    "content_preview": chunk.content[:200] + "...",
                    "speaker": chunk.speaker,
                    "timestamp": chunk.timestamp_start
                }
                for chunk in chunks[:5]  # Limit sources in response
            ]
            
            # Build response
            response = ChatResponse(
                response=response_text,
                conversation_id=request.conversation_id,
                sources=sources,
                metadata={
                    **retrieval_metadata,
                    "conversation_length": len(conversation_history)
                }
            )
            
            logger.info(
                "Chat request completed",
                conversation_id=request.conversation_id,
                chunks_used=len(chunks)
            )
            
            return response
            
        except Exception as e:
            chat_errors.inc()
            logger.error(
                "Chat request failed",
                error=str(e),
                conversation_id=request.conversation_id
            )
            raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/api/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat."""
    await websocket.accept()
    websocket_connections.inc()
    
    conversation_id = str(uuid.uuid4())
    conversation_history = []
    
    logger.info("WebSocket connection established", conversation_id=conversation_id)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            
            message = data.get("message", "")
            
            # Add to conversation history
            user_message = ChatMessage(
                role=MessageRole.USER,
                content=message
            )
            conversation_history.append(user_message)
            
            # Send acknowledgment
            await websocket.send_json({
                "type": "acknowledgment",
                "conversation_id": conversation_id
            })
            
            # Retrieve context
            chunks, retrieval_metadata = await rag_service.retrieve_context(
                query=message,
                conversation_history=conversation_history[:-1]
            )
            
            # Send sources
            await websocket.send_json({
                "type": "sources",
                "data": [
                    {
                        "meeting_id": chunk.meeting_id,
                        "content_preview": chunk.content[:100] + "..."
                    }
                    for chunk in chunks[:3]
                ]
            })
            
            # Format context
            context = await rag_service.format_context_for_llm(
                chunks=chunks,
                query=message
            )
            
            # Stream response
            await websocket.send_json({
                "type": "stream_start"
            })
            
            full_response = ""
            async for token in llm_service.generate_streaming_response(
                messages=conversation_history,
                context=context
            ):
                await websocket.send_json({
                    "type": "stream_token",
                    "token": token
                })
                full_response += token
            
            await websocket.send_json({
                "type": "stream_end"
            })
            
            # Add to history
            assistant_message = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=full_response
            )
            conversation_history.append(assistant_message)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected", conversation_id=conversation_id)
    except Exception as e:
        logger.error(
            "WebSocket error",
            error=str(e),
            conversation_id=conversation_id
        )
        await websocket.close()


@app.post("/api/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    """Search endpoint for direct knowledge base queries."""
    try:
        # Generate embedding for query
        query_embedding = await embeddings_service.get_embedding(request.query)
        
        if not query_embedding:
            raise HTTPException(status_code=400, detail="Failed to process query")
        
        # Perform search based on type
        if request.search_type == "vector":
            results = await vector_db.search(
                query_vector=query_embedding,
                top_k=request.top_k,
                filters=request.filters
            )
        elif request.search_type == "hybrid":
            results = await vector_db.hybrid_search(
                query_vector=query_embedding,
                query_text=request.query,
                top_k=request.top_k,
                filters=request.filters
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid search type")
        
        # Format results
        search_results = []
        for chunk, score in results:
            # Highlight matching text
            highlights = []
            query_terms = request.query.lower().split()
            for term in query_terms:
                if term in chunk.content.lower():
                    # Find and extract surrounding context
                    idx = chunk.content.lower().find(term)
                    start = max(0, idx - 30)
                    end = min(len(chunk.content), idx + len(term) + 30)
                    highlight = chunk.content[start:end]
                    if start > 0:
                        highlight = "..." + highlight
                    if end < len(chunk.content):
                        highlight = highlight + "..."
                    highlights.append(highlight)
            
            search_results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    highlights=highlights[:3]  # Limit highlights
                )
            )
        
        return search_results
        
    except Exception as e:
        logger.error("Search failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """Ingest new meeting data."""
    try:
        response = await ingestion_service.ingest_meeting(request)
        
        if response.status == "failed":
            raise HTTPException(status_code=400, detail=response.errors[0])
        
        return response
        
    except Exception as e:
        logger.error("Ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/bulk")
async def bulk_ingest(
    directory: str,
    file_pattern: str = "*.json"
):
    """Bulk ingest multiple files."""
    try:
        results = await ingestion_service.bulk_ingest(
            directory=directory,
            file_pattern=file_pattern
        )
        
        return results
        
    except Exception as e:
        logger.error("Bulk ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/meetings/{meeting_id}")
async def delete_meeting(meeting_id: str):
    """Delete a meeting and its chunks."""
    try:
        deleted = await vector_db.delete_meeting(meeting_id)
        
        return {
            "meeting_id": meeting_id,
            "deleted": deleted > 0
        }
        
    except Exception as e:
        logger.error(
            "Failed to delete meeting",
            meeting_id=meeting_id,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Get system statistics."""
    try:
        vector_stats = await vector_db.get_collection_stats()
        
        return {
            "vector_db": vector_stats,
            "conversations": {
                "active": len(conversations),
                "total_messages": sum(len(msgs) for msgs in conversations.values())
            }
        }
        
    except Exception as e:
        logger.error("Failed to get stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Add this import at the top
from backend.services.embeddings import embeddings_service


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
