"""Embeddings service for generating text embeddings."""

import asyncio
from typing import List, Dict, Optional, Any
import numpy as np
from openai import AsyncOpenAI
import tiktoken
import structlog
from aiocache import Cache
from aiocache.serializers import PickleSerializer
import hashlib

from backend.config import settings, MAX_RETRIES, RETRY_DELAY


logger = structlog.get_logger()


class EmbeddingsService:
    """Service for generating and managing text embeddings."""
    
    def __init__(self):
        """Initialize the embeddings service."""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model
        self.encoding = tiktoken.encoding_for_model(self.model)
        
        # Initialize cache
        self.cache = Cache(
            Cache.REDIS,
            endpoint=settings.redis_url.replace("redis://", "").split(":")[0],
            port=int(settings.redis_url.replace("redis://", "").split(":")[1]),
            ttl=settings.cache_ttl,
            serializer=PickleSerializer(),
            namespace="embeddings"
        )
        
        # Model-specific settings
        self.max_tokens = 8191  # Max tokens for text-embedding-3-small
        self.dimensions = EMBEDDING_DIMENSION
        
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for the text."""
        return hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()
    
    async def get_embedding(
        self, 
        text: str,
        use_cache: bool = True
    ) -> Optional[List[float]]:
        """Generate embedding for a single text."""
        if not text.strip():
            return None
        
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(text)
            cached = await self.cache.get(cache_key)
            if cached is not None:
                logger.debug("Retrieved embedding from cache", text_preview=text[:50])
                return cached
        
        # Truncate if too long
        tokens = self.encoding.encode(text)
        if len(tokens) > self.max_tokens:
            logger.warning(
                "Text exceeds token limit, truncating",
                original_tokens=len(tokens),
                max_tokens=self.max_tokens
            )
            tokens = tokens[:self.max_tokens]
            text = self.encoding.decode(tokens)
        
        # Generate embedding with retries
        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
                
                embedding = response.data[0].embedding
                
                # Cache the result
                if use_cache:
                    await self.cache.set(cache_key, embedding)
                
                return embedding
                
            except Exception as e:
                logger.error(
                    "Failed to generate embedding",
                    attempt=attempt + 1,
                    error=str(e)
                )
                
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    return None
        
        return None
    
    async def get_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 100,
        use_cache: bool = True
    ) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts efficiently."""
        if not texts:
            return []
        
        results = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache and prepare texts that need embedding
        for i, text in enumerate(texts):
            if not text.strip():
                continue
                
            if use_cache:
                cache_key = self._get_cache_key(text)
                cached = await self.cache.get(cache_key)
                if cached is not None:
                    results[i] = cached
                    continue
            
            texts_to_embed.append(text)
            indices_to_embed.append(i)
        
        # Batch process texts that need embedding
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i + batch_size]
            batch_indices = indices_to_embed[i:i + batch_size]
            
            # Truncate texts if needed
            processed_texts = []
            for text in batch_texts:
                tokens = self.encoding.encode(text)
                if len(tokens) > self.max_tokens:
                    tokens = tokens[:self.max_tokens]
                    text = self.encoding.decode(tokens)
                processed_texts.append(text)
            
            # Generate embeddings for batch
            for attempt in range(MAX_RETRIES):
                try:
                    response = await self.client.embeddings.create(
                        input=processed_texts,
                        model=self.model
                    )
                    
                    # Process results
                    for j, (idx, text) in enumerate(zip(batch_indices, batch_texts)):
                        embedding = response.data[j].embedding
                        results[idx] = embedding
                        
                        # Cache the result
                        if use_cache:
                            cache_key = self._get_cache_key(text)
                            await self.cache.set(cache_key, embedding)
                    
                    break
                    
                except Exception as e:
                    logger.error(
                        "Failed to generate batch embeddings",
                        batch_start=i,
                        batch_size=len(batch_texts),
                        attempt=attempt + 1,
                        error=str(e)
                    )
                    
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))
        
        logger.info(
            "Generated embeddings batch",
            total_texts=len(texts),
            embedded=len([r for r in results if r is not None]),
            cached=len(texts) - len(texts_to_embed)
        )
        
        return results
    
    async def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    async def find_similar_embeddings(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[tuple[int, float]]:
        """Find most similar embeddings from candidates."""
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = await self.compute_similarity(query_embedding, candidate)
            if similarity >= threshold:
                similarities.append((i, similarity))
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text."""
        return len(self.encoding.encode(text))
    
    async def clear_cache(self) -> None:
        """Clear the embeddings cache."""
        await self.cache.clear()
        logger.info("Cleared embeddings cache")
    
    async def warm_cache(self, texts: List[str]) -> int:
        """Pre-compute and cache embeddings for frequently used texts."""
        embeddings = await self.get_embeddings_batch(texts, use_cache=True)
        cached_count = len([e for e in embeddings if e is not None])
        
        logger.info(
            "Warmed embeddings cache",
            total_texts=len(texts),
            cached=cached_count
        )
        
        return cached_count


# Global instance
embeddings_service = EmbeddingsService()
