"""Vector database service for managing embeddings and similarity search."""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, Range, MatchValue,
    SearchRequest, ScoredPoint, UpdateStatus
)
from qdrant_client.http.exceptions import UnexpectedResponse
import structlog
from datetime import datetime

from backend.config import settings, EMBEDDING_DIMENSION
from backend.models import TranscriptionChunk


logger = structlog.get_logger()


class VectorDBService:
    """Service for managing vector database operations."""
    
    def __init__(self):
        """Initialize the vector database connection."""
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            timeout=30
        )
        self.collection_name = settings.qdrant_collection_name
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the collection if it doesn't exist."""
        if self._initialized:
            return
            
        try:
            # Check if collection exists
            collections = await asyncio.to_thread(
                self.client.get_collections
            )
            
            exists = any(
                col.name == self.collection_name 
                for col in collections.collections
            )
            
            if not exists:
                # Create collection with vector configuration
                await asyncio.to_thread(
                    self.client.create_collection,
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIMENSION,
                        distance=Distance.COSINE
                    )
                )
                
                # Create indexes for metadata fields
                await self._create_indexes()
                
                logger.info(
                    "Created vector collection",
                    collection=self.collection_name
                )
            else:
                logger.info(
                    "Vector collection already exists",
                    collection=self.collection_name
                )
                
            self._initialized = True
            
        except Exception as e:
            logger.error(
                "Failed to initialize vector database",
                error=str(e)
            )
            raise
    
    async def _create_indexes(self) -> None:
        """Create indexes for efficient filtering."""
        # Note: Qdrant automatically indexes payload fields
        # This is a placeholder for any custom index configuration
        pass
    
    async def upsert_chunks(
        self, 
        chunks: List[TranscriptionChunk],
        batch_size: int = 100
    ) -> int:
        """Insert or update transcription chunks with embeddings."""
        if not chunks:
            return 0
            
        await self.initialize()
        
        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning(
                    "Skipping chunk without embedding",
                    chunk_id=chunk.id
                )
                continue
                
            # Prepare payload
            payload = {
                "meeting_id": chunk.meeting_id,
                "content": chunk.content,
                "timestamp_start": chunk.timestamp_start,
                "timestamp_end": chunk.timestamp_end,
                "speaker": chunk.speaker,
                "metadata": chunk.metadata,
                "indexed_at": datetime.utcnow().isoformat()
            }
            
            # Create point
            point = PointStruct(
                id=chunk.id,
                vector=chunk.embedding,
                payload=payload
            )
            points.append(point)
        
        # Batch upsert
        total_upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            
            try:
                result = await asyncio.to_thread(
                    self.client.upsert,
                    collection_name=self.collection_name,
                    points=batch
                )
                
                if result.status == UpdateStatus.COMPLETED:
                    total_upserted += len(batch)
                    
            except Exception as e:
                logger.error(
                    "Failed to upsert batch",
                    batch_start=i,
                    batch_size=len(batch),
                    error=str(e)
                )
                
        logger.info(
            "Upserted chunks to vector database",
            total_chunks=len(chunks),
            successful=total_upserted
        )
        
        return total_upserted
    
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[TranscriptionChunk, float]]:
        """Perform similarity search with optional filtering."""
        await self.initialize()
        
        # Build filter conditions
        filter_conditions = []
        
        if filters:
            # Meeting ID filter
            if "meeting_id" in filters:
                filter_conditions.append(
                    FieldCondition(
                        key="meeting_id",
                        match=MatchValue(value=filters["meeting_id"])
                    )
                )
            
            # Date range filter
            if "date_range" in filters:
                # This would require timestamp fields in the payload
                pass
            
            # Speaker filter
            if "speaker" in filters:
                filter_conditions.append(
                    FieldCondition(
                        key="speaker",
                        match=MatchValue(value=filters["speaker"])
                    )
                )
        
        # Construct search request
        search_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        try:
            # Perform search
            results = await asyncio.to_thread(
                self.client.search,
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=search_filter,
                score_threshold=score_threshold
            )
            
            # Convert results to chunks
            chunk_results = []
            for point in results:
                chunk = self._point_to_chunk(point)
                if chunk:
                    chunk_results.append((chunk, point.score))
            
            return chunk_results
            
        except Exception as e:
            logger.error(
                "Vector search failed",
                error=str(e)
            )
            return []
    
    async def hybrid_search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[TranscriptionChunk, float]]:
        """Perform hybrid search combining vector and keyword search."""
        await self.initialize()
        
        # Vector search
        vector_results = await self.search(
            query_vector=query_vector,
            top_k=top_k * 2,  # Get more results for merging
            filters=filters
        )
        
        # Convert to dict for easy lookup
        vector_scores = {chunk.id: score for chunk, score in vector_results}
        
        # Keyword search using Qdrant's scroll with text matching
        # Note: This is a simplified version. In production, you might want
        # to use a dedicated text search engine like Elasticsearch
        keyword_results = await self._keyword_search(
            query_text=query_text,
            limit=top_k * 2,
            filters=filters
        )
        
        # Merge results with weighted scoring
        combined_scores = {}
        all_chunks = {}
        
        # Add vector search results
        for chunk, score in vector_results:
            combined_scores[chunk.id] = score * vector_weight
            all_chunks[chunk.id] = chunk
        
        # Add keyword search results
        for chunk, score in keyword_results:
            if chunk.id in combined_scores:
                combined_scores[chunk.id] += score * (1 - vector_weight)
            else:
                combined_scores[chunk.id] = score * (1 - vector_weight)
                all_chunks[chunk.id] = chunk
        
        # Sort by combined score and return top k
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [(all_chunks[chunk_id], score) for chunk_id, score in sorted_results]
    
    async def _keyword_search(
        self,
        query_text: str,
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[TranscriptionChunk, float]]:
        """Perform keyword-based search."""
        # This is a simplified implementation
        # In production, consider using Qdrant's full-text search or external service
        
        # For now, we'll do a simple scroll through all points and score by keyword matches
        # This is not efficient for large datasets
        
        try:
            all_points = []
            offset = None
            
            while True:
                result = await asyncio.to_thread(
                    self.client.scroll,
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                points, offset = result
                all_points.extend(points)
                
                if offset is None:
                    break
            
            # Score by keyword matches
            query_terms = query_text.lower().split()
            scored_results = []
            
            for point in all_points:
                content = point.payload.get("content", "").lower()
                
                # Simple scoring: count matching terms
                score = sum(1 for term in query_terms if term in content)
                
                if score > 0:
                    chunk = self._point_to_chunk(point)
                    if chunk:
                        # Normalize score
                        normalized_score = score / len(query_terms)
                        scored_results.append((chunk, normalized_score))
            
            # Sort by score and return top results
            scored_results.sort(key=lambda x: x[1], reverse=True)
            return scored_results[:limit]
            
        except Exception as e:
            logger.error(
                "Keyword search failed",
                error=str(e)
            )
            return []
    
    def _point_to_chunk(self, point: ScoredPoint) -> Optional[TranscriptionChunk]:
        """Convert a Qdrant point to a TranscriptionChunk."""
        try:
            return TranscriptionChunk(
                id=str(point.id),
                meeting_id=point.payload["meeting_id"],
                content=point.payload["content"],
                timestamp_start=point.payload.get("timestamp_start"),
                timestamp_end=point.payload.get("timestamp_end"),
                speaker=point.payload.get("speaker"),
                metadata=point.payload.get("metadata", {})
            )
        except Exception as e:
            logger.error(
                "Failed to convert point to chunk",
                point_id=point.id,
                error=str(e)
            )
            return None
    
    async def delete_meeting(self, meeting_id: str) -> int:
        """Delete all chunks for a specific meeting."""
        await self.initialize()
        
        try:
            result = await asyncio.to_thread(
                self.client.delete,
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="meeting_id",
                            match=MatchValue(value=meeting_id)
                        )
                    ]
                )
            )
            
            logger.info(
                "Deleted meeting chunks",
                meeting_id=meeting_id
            )
            
            return 1  # Qdrant doesn't return count
            
        except Exception as e:
            logger.error(
                "Failed to delete meeting",
                meeting_id=meeting_id,
                error=str(e)
            )
            return 0
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection."""
        await self.initialize()
        
        try:
            info = await asyncio.to_thread(
                self.client.get_collection,
                collection_name=self.collection_name
            )
            
            return {
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status
            }
            
        except Exception as e:
            logger.error(
                "Failed to get collection stats",
                error=str(e)
            )
            return {}


# Global instance
vector_db = VectorDBService()
