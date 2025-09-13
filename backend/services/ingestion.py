"""Data ingestion service for processing meeting transcriptions and audio files."""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from pathlib import Path
import re
from datetime import datetime
import structlog
import numpy as np
from pydub import AudioSegment
import librosa
import soundfile as sf

from backend.config import settings
from backend.models import TranscriptionChunk, Meeting, IngestRequest, IngestResponse
from backend.services.embeddings import embeddings_service
from backend.services.vector_db import vector_db
from backend.services.chunking import ChunkingService


logger = structlog.get_logger()


class IngestionService:
    """Service for ingesting meeting transcriptions and audio files."""
    
    def __init__(self):
        """Initialize the ingestion service."""
        self.chunking_service = ChunkingService()
        
    async def ingest_meeting(
        self,
        request: IngestRequest
    ) -> IngestResponse:
        """Ingest a meeting's transcription and/or audio data."""
        start_time = asyncio.get_event_loop().time()
        errors = []
        chunks_created = 0
        
        try:
            # Process transcription
            if request.transcription_text or request.transcription_file_path:
                transcription_chunks = await self._process_transcription(
                    meeting_id=request.meeting_id,
                    text=request.transcription_text,
                    file_path=request.transcription_file_path,
                    metadata=request.metadata
                )
                chunks_created += len(transcription_chunks)
            
            # Process audio (if provided)
            if request.audio_file_path:
                audio_chunks = await self._process_audio(
                    meeting_id=request.meeting_id,
                    file_path=request.audio_file_path,
                    metadata=request.metadata
                )
                chunks_created += len(audio_chunks)
            
            # Calculate processing time
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return IngestResponse(
                meeting_id=request.meeting_id,
                chunks_created=chunks_created,
                processing_time_ms=processing_time,
                status="success" if not errors else "partial",
                errors=errors if errors else None
            )
            
        except Exception as e:
            logger.error(
                "Failed to ingest meeting",
                meeting_id=request.meeting_id,
                error=str(e)
            )
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return IngestResponse(
                meeting_id=request.meeting_id,
                chunks_created=chunks_created,
                processing_time_ms=processing_time,
                status="failed",
                errors=[str(e)]
            )
    
    async def _process_transcription(
        self,
        meeting_id: str,
        text: Optional[str] = None,
        file_path: Optional[str] = None,
        metadata: Dict[str, Any] = {}
    ) -> List[TranscriptionChunk]:
        """Process transcription text or file."""
        # Load transcription text
        if text:
            transcription_text = text
        elif file_path:
            transcription_text = await self._load_transcription_file(file_path)
        else:
            raise ValueError("Either text or file_path must be provided")
        
        # Parse transcription format (could be various formats)
        parsed_segments = await self._parse_transcription(transcription_text)
        
        # Create chunks
        chunks = []
        for segment in parsed_segments:
            # Split large segments into smaller chunks
            segment_chunks = await self.chunking_service.create_chunks(
                text=segment["content"],
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
                metadata={
                    "speaker": segment.get("speaker"),
                    "timestamp_start": segment.get("timestamp_start"),
                    "timestamp_end": segment.get("timestamp_end"),
                    **metadata
                }
            )
            
            # Create TranscriptionChunk objects
            for i, chunk_text in enumerate(segment_chunks):
                chunk_id = f"{meeting_id}_seg{segment.get('index', 0)}_chunk{i}"
                
                chunk = TranscriptionChunk(
                    id=chunk_id,
                    meeting_id=meeting_id,
                    content=chunk_text,
                    timestamp_start=segment.get("timestamp_start"),
                    timestamp_end=segment.get("timestamp_end"),
                    speaker=segment.get("speaker"),
                    metadata={
                        **metadata,
                        "segment_index": segment.get("index", 0),
                        "chunk_index": i
                    }
                )
                chunks.append(chunk)
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = await embeddings_service.get_embeddings_batch(texts)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        # Store in vector database
        successful_chunks = [c for c in chunks if c.embedding is not None]
        await vector_db.upsert_chunks(successful_chunks)
        
        logger.info(
            "Processed transcription",
            meeting_id=meeting_id,
            total_chunks=len(chunks),
            successful=len(successful_chunks)
        )
        
        return successful_chunks
    
    async def _load_transcription_file(self, file_path: str) -> str:
        """Load transcription from file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Transcription file not found: {file_path}")
        
        # Support various formats
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Extract text based on common formats
            if isinstance(data, dict):
                # Krisp format might have transcript key
                if "transcript" in data:
                    return data["transcript"]
                elif "text" in data:
                    return data["text"]
                elif "segments" in data:
                    # Join segments
                    segments = data["segments"]
                    return "\n".join(
                        seg.get("text", "") for seg in segments
                    )
            elif isinstance(data, list):
                # List of segments
                return "\n".join(
                    seg.get("text", "") if isinstance(seg, dict) else str(seg)
                    for seg in data
                )
                
            return json.dumps(data)  # Fallback
            
        else:
            # Plain text file
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    
    async def _parse_transcription(
        self, 
        text: str
    ) -> List[Dict[str, Any]]:
        """Parse transcription text into segments."""
        segments = []
        
        # Try to detect format
        # Format 1: Speaker labels (e.g., "John: Hello everyone")
        speaker_pattern = re.compile(r"^([^:]+):\s*(.+)$", re.MULTILINE)
        
        # Format 2: Timestamps [00:00:00] or (00:00:00)
        timestamp_pattern = re.compile(
            r"[\[\(](\d{2}:\d{2}:\d{2})[\]\)]\s*(?:([^:]+):\s*)?(.+)$",
            re.MULTILINE
        )
        
        # Try timestamp format first
        timestamp_matches = list(timestamp_pattern.finditer(text))
        if timestamp_matches:
            for i, match in enumerate(timestamp_matches):
                timestamp_str, speaker, content = match.groups()
                
                # Convert timestamp to seconds
                parts = timestamp_str.split(":")
                timestamp = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                
                segments.append({
                    "index": i,
                    "content": content.strip(),
                    "speaker": speaker.strip() if speaker else None,
                    "timestamp_start": float(timestamp),
                    "timestamp_end": None  # Will be set from next segment
                })
            
            # Set end timestamps
            for i in range(len(segments) - 1):
                segments[i]["timestamp_end"] = segments[i + 1]["timestamp_start"]
                
        else:
            # Try speaker format
            speaker_matches = list(speaker_pattern.finditer(text))
            if speaker_matches:
                for i, match in enumerate(speaker_matches):
                    speaker, content = match.groups()
                    segments.append({
                        "index": i,
                        "content": content.strip(),
                        "speaker": speaker.strip(),
                        "timestamp_start": None,
                        "timestamp_end": None
                    })
            else:
                # Plain text - treat as single segment
                segments.append({
                    "index": 0,
                    "content": text.strip(),
                    "speaker": None,
                    "timestamp_start": None,
                    "timestamp_end": None
                })
        
        return segments
    
    async def _process_audio(
        self,
        meeting_id: str,
        file_path: str,
        metadata: Dict[str, Any] = {}
    ) -> List[TranscriptionChunk]:
        """Process audio file for future audio-based features."""
        # This is a placeholder for audio processing
        # In a real implementation, you might:
        # 1. Extract audio features (MFCCs, spectrograms)
        # 2. Perform speaker diarization
        # 3. Generate audio embeddings
        # 4. Align with transcription if available
        
        logger.info(
            "Audio processing placeholder",
            meeting_id=meeting_id,
            audio_file=file_path
        )
        
        # For now, just return empty list
        # This makes it easy to add audio processing later
        return []
    
    async def bulk_ingest(
        self,
        directory: str,
        file_pattern: str = "*.json",
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """Bulk ingest multiple meeting files from a directory."""
        path = Path(directory)
        
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Directory not found: {directory}")
        
        # Find all matching files
        files = list(path.glob(file_pattern))
        
        logger.info(
            "Starting bulk ingestion",
            directory=directory,
            files_found=len(files)
        )
        
        results = {
            "total_files": len(files),
            "successful": 0,
            "failed": 0,
            "total_chunks": 0,
            "errors": []
        }
        
        # Process in batches
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            
            # Process batch concurrently
            tasks = []
            for file_path in batch_files:
                # Generate meeting ID from filename
                meeting_id = file_path.stem
                
                request = IngestRequest(
                    meeting_id=meeting_id,
                    transcription_file_path=str(file_path),
                    metadata={
                        "source_file": file_path.name,
                        "ingested_at": datetime.utcnow().isoformat()
                    }
                )
                
                task = self.ingest_meeting(request)
                tasks.append(task)
            
            # Wait for batch to complete
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for file_path, response in zip(batch_files, responses):
                if isinstance(response, Exception):
                    results["failed"] += 1
                    results["errors"].append({
                        "file": str(file_path),
                        "error": str(response)
                    })
                elif response.status == "success":
                    results["successful"] += 1
                    results["total_chunks"] += response.chunks_created
                else:
                    results["failed"] += 1
                    if response.errors:
                        results["errors"].append({
                            "file": str(file_path),
                            "errors": response.errors
                        })
        
        logger.info(
            "Bulk ingestion completed",
            **results
        )
        
        return results


class ChunkingService:
    """Service for creating text chunks with various strategies."""
    
    async def create_chunks(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200,
        metadata: Dict[str, Any] = {}
    ) -> List[str]:
        """Create overlapping chunks from text."""
        if not text:
            return []
        
        chunks = []
        text_length = len(text)
        
        # Handle case where text is shorter than chunk size
        if text_length <= chunk_size:
            return [text]
        
        # Create overlapping chunks
        start = 0
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # Try to find a good break point (sentence end)
            if end < text_length:
                # Look for sentence endings
                break_points = [
                    text.rfind(". ", start, end),
                    text.rfind("! ", start, end),
                    text.rfind("? ", start, end),
                    text.rfind("\n", start, end)
                ]
                
                # Use the latest break point if found
                best_break = max(break_points)
                if best_break > start:
                    end = best_break + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position
            start = end - overlap if end < text_length else end
        
        return chunks
    
    async def create_semantic_chunks(
        self,
        text: str,
        max_chunk_size: int = 1500,
        similarity_threshold: float = 0.7
    ) -> List[str]:
        """Create chunks based on semantic similarity (advanced method)."""
        # This is a placeholder for semantic chunking
        # In production, you might use:
        # 1. Sentence embeddings to find semantic boundaries
        # 2. Topic modeling to group related content
        # 3. Sliding window with similarity scoring
        
        # For now, fall back to regular chunking
        return await self.create_chunks(text, max_chunk_size)


# Global instance
ingestion_service = IngestionService()
