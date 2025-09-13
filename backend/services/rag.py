"""RAG (Retrieval-Augmented Generation) service for intelligent context retrieval."""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import structlog
import tiktoken
from collections import defaultdict

from backend.config import settings
from backend.models import TranscriptionChunk, ChatMessage, MessageRole
from backend.services.embeddings import embeddings_service
from backend.services.vector_db import vector_db
from backend.services.llm import llm_service


logger = structlog.get_logger()


class RAGService:
    """Advanced RAG service with multi-stage retrieval and reranking."""
    
    def __init__(self):
        """Initialize the RAG service."""
        self.encoding = tiktoken.encoding_for_model(settings.openai_model)
        
    async def retrieve_context(
        self,
        query: str,
        conversation_history: Optional[List[ChatMessage]] = None,
        filters: Optional[Dict[str, Any]] = None,
        max_context_tokens: Optional[int] = None
    ) -> Tuple[List[TranscriptionChunk], Dict[str, Any]]:
        """Retrieve relevant context using multi-stage RAG pipeline."""
        start_time = asyncio.get_event_loop().time()
        
        # Use provided max tokens or default from settings
        max_tokens = max_context_tokens or (settings.max_context_length // 2)
        
        # Enhance query with conversation context
        enhanced_query = await self._enhance_query(query, conversation_history)
        
        # Generate query embedding
        query_embedding = await embeddings_service.get_embedding(enhanced_query)
        if query_embedding is None:
            logger.error("Failed to generate query embedding")
            return [], {"error": "Failed to generate query embedding"}
        
        # Stage 1: Initial retrieval (cast a wide net)
        initial_results = await vector_db.hybrid_search(
            query_vector=query_embedding,
            query_text=query,
            top_k=settings.top_k_retrieval,
            vector_weight=settings.hybrid_search_weight,
            filters=filters
        )
        
        if not initial_results:
            logger.info("No results found for query", query=query)
            return [], {"stages": {"initial_retrieval": 0}}
        
        # Stage 2: Reranking based on relevance
        reranked_results = await self._rerank_results(
            query=enhanced_query,
            results=initial_results,
            top_k=settings.top_k_rerank * 2  # Keep more for diversity
        )
        
        # Stage 3: Diversity and deduplication
        diverse_results = await self._ensure_diversity(
            results=reranked_results,
            top_k=settings.top_k_rerank
        )
        
        # Stage 4: Context window optimization
        final_chunks = await self._optimize_context_window(
            chunks=[chunk for chunk, _ in diverse_results],
            max_tokens=max_tokens
        )
        
        # Calculate metrics
        retrieval_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        metadata = {
            "stages": {
                "initial_retrieval": len(initial_results),
                "after_reranking": len(reranked_results),
                "after_diversity": len(diverse_results),
                "final_chunks": len(final_chunks)
            },
            "retrieval_time_ms": retrieval_time,
            "total_tokens": sum(
                len(self.encoding.encode(chunk.content)) 
                for chunk in final_chunks
            )
        }
        
        logger.info(
            "RAG retrieval completed",
            query_preview=query[:50],
            **metadata
        )
        
        return final_chunks, metadata
    
    async def _enhance_query(
        self,
        query: str,
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> str:
        """Enhance query with conversation context for better retrieval."""
        if not conversation_history:
            return query
        
        # Extract key context from recent messages
        recent_context = []
        for msg in conversation_history[-3:]:  # Last 3 messages
            if msg.role == MessageRole.USER:
                recent_context.append(f"User asked: {msg.content[:100]}")
            elif msg.role == MessageRole.ASSISTANT:
                # Extract key points from assistant responses
                if "meeting" in msg.content.lower():
                    recent_context.append("Context: discussing meetings")
                if "action" in msg.content.lower():
                    recent_context.append("Context: discussing action items")
        
        # Combine with original query
        if recent_context:
            enhanced = f"{query}\n\nConversation context: {'; '.join(recent_context)}"
            return enhanced
        
        return query
    
    async def _rerank_results(
        self,
        query: str,
        results: List[Tuple[TranscriptionChunk, float]],
        top_k: int
    ) -> List[Tuple[TranscriptionChunk, float]]:
        """Rerank results using advanced scoring."""
        reranked = []
        
        for chunk, base_score in results:
            # Calculate additional scoring factors
            
            # 1. Keyword matching score
            query_terms = set(query.lower().split())
            chunk_terms = set(chunk.content.lower().split())
            keyword_overlap = len(query_terms & chunk_terms) / len(query_terms)
            
            # 2. Recency score (if timestamp available)
            recency_score = 1.0
            if chunk.metadata.get("date"):
                try:
                    chunk_date = datetime.fromisoformat(chunk.metadata["date"])
                    days_old = (datetime.utcnow() - chunk_date).days
                    recency_score = 1.0 / (1.0 + days_old / 30.0)  # Decay over 30 days
                except:
                    pass
            
            # 3. Length penalty (prefer substantial chunks)
            length_score = min(len(chunk.content) / 500.0, 1.0)
            
            # 4. Speaker diversity bonus (if tracking speakers)
            speaker_bonus = 0.1 if chunk.speaker else 0.0
            
            # Combine scores
            final_score = (
                base_score * 0.6 +  # Original similarity
                keyword_overlap * 0.2 +
                recency_score * 0.1 +
                length_score * 0.05 +
                speaker_bonus * 0.05
            )
            
            reranked.append((chunk, final_score))
        
        # Sort by final score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked[:top_k]
    
    async def _ensure_diversity(
        self,
        results: List[Tuple[TranscriptionChunk, float]],
        top_k: int,
        similarity_threshold: float = 0.85
    ) -> List[Tuple[TranscriptionChunk, float]]:
        """Ensure diversity in results by removing near-duplicates."""
        if not results:
            return []
        
        diverse_results = [results[0]]  # Always keep the top result
        
        for chunk, score in results[1:]:
            # Check similarity with already selected chunks
            is_diverse = True
            
            for selected_chunk, _ in diverse_results:
                # Quick content similarity check
                if chunk.meeting_id == selected_chunk.meeting_id:
                    # Same meeting - check for overlap
                    overlap = self._calculate_text_overlap(
                        chunk.content,
                        selected_chunk.content
                    )
                    
                    if overlap > similarity_threshold:
                        is_diverse = False
                        break
            
            if is_diverse:
                diverse_results.append((chunk, score))
                
            if len(diverse_results) >= top_k:
                break
        
        return diverse_results
    
    def _calculate_text_overlap(self, text1: str, text2: str) -> float:
        """Calculate overlap between two texts."""
        # Simple word-based overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    async def _optimize_context_window(
        self,
        chunks: List[TranscriptionChunk],
        max_tokens: int
    ) -> List[TranscriptionChunk]:
        """Optimize chunks to fit within context window."""
        if not chunks:
            return []
        
        selected_chunks = []
        total_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = len(self.encoding.encode(chunk.content))
            
            if total_tokens + chunk_tokens <= max_tokens:
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
            else:
                # Try to fit a truncated version
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 100:  # Minimum useful chunk size
                    truncated_content = self._truncate_to_tokens(
                        chunk.content,
                        remaining_tokens
                    )
                    
                    truncated_chunk = TranscriptionChunk(
                        **chunk.dict(),
                        content=truncated_content + " [truncated]"
                    )
                    selected_chunks.append(truncated_chunk)
                    
                break
        
        return selected_chunks
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate and decode
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
    
    async def format_context_for_llm(
        self,
        chunks: List[TranscriptionChunk],
        query: str
    ) -> str:
        """Format retrieved chunks into context for LLM."""
        if not chunks:
            return "No relevant context found for the query."
        
        # Group chunks by meeting
        chunks_by_meeting = defaultdict(list)
        for chunk in chunks:
            chunks_by_meeting[chunk.meeting_id].append(chunk)
        
        # Format context
        context_parts = []
        
        for meeting_id, meeting_chunks in chunks_by_meeting.items():
            # Sort by timestamp if available
            meeting_chunks.sort(
                key=lambda c: (c.timestamp_start or 0, c.chunk_index or 0)
            )
            
            # Add meeting header
            meeting_info = meeting_chunks[0].metadata
            header = f"\n--- Meeting: {meeting_info.get('title', meeting_id)} ---"
            
            if meeting_info.get('date'):
                header += f"\nDate: {meeting_info['date']}"
            
            if meeting_info.get('participants'):
                header += f"\nParticipants: {', '.join(meeting_info['participants'])}"
            
            context_parts.append(header)
            
            # Add chunks
            for chunk in meeting_chunks:
                chunk_text = chunk.content
                
                # Add speaker info if available
                if chunk.speaker:
                    chunk_text = f"[{chunk.speaker}]: {chunk_text}"
                
                # Add timestamp if available
                if chunk.timestamp_start is not None:
                    timestamp = self._format_timestamp(chunk.timestamp_start)
                    chunk_text = f"[{timestamp}] {chunk_text}"
                
                context_parts.append(chunk_text)
        
        # Combine all parts
        full_context = "\n\n".join(context_parts)
        
        # Add query context
        return f"""Based on the following meeting transcription excerpts, please answer the user's question.

Meeting Context:
{full_context}

User Question: {query}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please indicate what information is missing."""
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp in HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    async def extract_action_items(
        self,
        chunks: List[TranscriptionChunk]
    ) -> List[Dict[str, Any]]:
        """Extract action items from meeting chunks."""
        # This could be enhanced with NLP or LLM-based extraction
        action_items = []
        
        action_keywords = [
            "action item", "todo", "to do", "will", "need to",
            "should", "must", "assign", "responsible", "deadline"
        ]
        
        for chunk in chunks:
            content_lower = chunk.content.lower()
            
            # Check for action keywords
            if any(keyword in content_lower for keyword in action_keywords):
                action_items.append({
                    "content": chunk.content,
                    "speaker": chunk.speaker,
                    "meeting_id": chunk.meeting_id,
                    "timestamp": chunk.timestamp_start
                })
        
        return action_items


# Global instance
rag_service = RAGService()
