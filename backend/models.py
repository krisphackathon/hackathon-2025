"""Data models for the application."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Chat message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class TranscriptionChunk(BaseModel):
    """Model for a chunk of transcription data."""
    id: str = Field(..., description="Unique identifier for the chunk")
    meeting_id: str = Field(..., description="ID of the source meeting")
    content: str = Field(..., description="The actual transcription text")
    timestamp_start: Optional[float] = Field(None, description="Start timestamp in seconds")
    timestamp_end: Optional[float] = Field(None, description="End timestamp in seconds")
    speaker: Optional[str] = Field(None, description="Speaker identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the content")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "chunk_123",
                "meeting_id": "meeting_456",
                "content": "Let's discuss the Q4 roadmap...",
                "timestamp_start": 120.5,
                "timestamp_end": 145.8,
                "speaker": "John Doe",
                "metadata": {"department": "Engineering", "topic": "Planning"}
            }
        }


class Meeting(BaseModel):
    """Model for meeting data."""
    id: str = Field(..., description="Unique meeting identifier")
    title: str = Field(..., description="Meeting title")
    date: datetime = Field(..., description="Meeting date and time")
    duration: Optional[float] = Field(None, description="Duration in seconds")
    participants: List[str] = Field(default_factory=list, description="List of participants")
    summary: Optional[str] = Field(None, description="AI-generated meeting summary")
    action_items: List[str] = Field(default_factory=list, description="Extracted action items")
    tags: List[str] = Field(default_factory=list, description="Meeting tags/topics")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ChatMessage(BaseModel):
    """Model for chat messages."""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User's chat message")
    conversation_id: Optional[str] = Field(None, description="ID for conversation continuity")
    context_window: Optional[int] = Field(None, description="Override default context window")
    search_filters: Optional[Dict[str, Any]] = Field(None, description="Filters for search")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What were the main decisions from the product meeting last week?",
                "conversation_id": "conv_789",
                "search_filters": {"date_range": "last_week", "participants": ["John", "Jane"]}
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="AI assistant's response")
    conversation_id: str = Field(..., description="Conversation ID for continuity")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source chunks used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Based on the product meeting transcript from last week...",
                "conversation_id": "conv_789",
                "sources": [
                    {
                        "meeting_id": "meeting_456",
                        "chunk_id": "chunk_123",
                        "relevance_score": 0.92
                    }
                ],
                "metadata": {
                    "tokens_used": 1250,
                    "retrieval_time_ms": 145,
                    "generation_time_ms": 2300
                }
            }
        }


class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(10, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    search_type: str = Field("hybrid", description="Search type: vector, keyword, or hybrid")


class SearchResult(BaseModel):
    """Model for search results."""
    chunk: TranscriptionChunk
    score: float = Field(..., description="Relevance score")
    highlights: Optional[List[str]] = Field(None, description="Highlighted text snippets")


class IngestRequest(BaseModel):
    """Request model for data ingestion."""
    meeting_id: str = Field(..., description="Meeting identifier")
    transcription_text: Optional[str] = Field(None, description="Full transcription text")
    transcription_file_path: Optional[str] = Field(None, description="Path to transcription file")
    audio_file_path: Optional[str] = Field(None, description="Path to audio file")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Meeting metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "meeting_id": "meeting_789",
                "transcription_text": "Meeting transcript content...",
                "metadata": {
                    "title": "Q4 Planning Meeting",
                    "date": "2025-09-10T14:00:00Z",
                    "participants": ["John Doe", "Jane Smith"]
                }
            }
        }


class IngestResponse(BaseModel):
    """Response model for data ingestion."""
    meeting_id: str
    chunks_created: int
    processing_time_ms: float
    status: str = Field(..., description="Status: success, partial, or failed")
    errors: Optional[List[str]] = None
