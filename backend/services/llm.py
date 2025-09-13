"""LLM service for generating responses using OpenAI."""

import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
import json
from openai import AsyncOpenAI
import tiktoken
import structlog
from aiocache import Cache
from aiocache.serializers import JsonSerializer
import hashlib

from backend.config import settings, MAX_RETRIES, RETRY_DELAY
from backend.models import ChatMessage, MessageRole


logger = structlog.get_logger()


class LLMService:
    """Service for LLM interactions with streaming support."""
    
    def __init__(self):
        """Initialize the LLM service."""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.encoding = tiktoken.encoding_for_model(self.model)
        
        # Initialize cache for responses
        self.cache = Cache(
            Cache.REDIS,
            endpoint=settings.redis_url.replace("redis://", "").split(":")[0],
            port=int(settings.redis_url.replace("redis://", "").split(":")[1]),
            ttl=settings.cache_ttl,
            serializer=JsonSerializer(),
            namespace="llm_responses"
        )
        
        # System prompt for meeting assistant
        self.system_prompt = """You are an AI assistant specialized in analyzing meeting transcriptions. 
Your role is to help users understand meeting content, extract insights, find action items, and answer questions about discussions.

Key capabilities:
- Summarize meeting discussions and decisions
- Extract and track action items and commitments
- Identify key topics and themes
- Find specific information across multiple meetings
- Provide context about participants and their contributions

Always base your responses on the provided meeting context. If information is not available in the context, clearly state that.
Be concise but comprehensive in your answers."""
    
    def _get_cache_key(self, messages: List[ChatMessage], context: str) -> str:
        """Generate a cache key for the conversation."""
        # Include last few messages and context in cache key
        key_parts = [
            self.model,
            context[:500],  # First 500 chars of context
            *[f"{m.role}:{m.content[:100]}" for m in messages[-3:]]  # Last 3 messages
        ]
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def generate_response(
        self,
        messages: List[ChatMessage],
        context: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        use_cache: bool = True
    ) -> str:
        """Generate a response for the conversation."""
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(messages, context)
            cached = await self.cache.get(cache_key)
            if cached:
                logger.debug("Retrieved response from cache")
                return cached
        
        # Prepare messages for API
        api_messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add context as first user message
        api_messages.append({
            "role": "user",
            "content": context
        })
        
        # Add conversation history
        for msg in messages:
            api_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        # Calculate token budget
        prompt_tokens = sum(
            len(self.encoding.encode(msg["content"]))
            for msg in api_messages
        )
        
        if max_tokens is None:
            # Leave room for response
            max_tokens = min(
                4000,  # Reasonable response limit
                settings.max_context_length - prompt_tokens - 100
            )
        
        # Generate response with retries
        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=api_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    presence_penalty=0.1,
                    frequency_penalty=0.1
                )
                
                content = response.choices[0].message.content
                
                # Cache the response
                if use_cache and content:
                    await self.cache.set(cache_key, content)
                
                logger.info(
                    "Generated LLM response",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )
                
                return content
                
            except Exception as e:
                logger.error(
                    "Failed to generate response",
                    attempt=attempt + 1,
                    error=str(e)
                )
                
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    return "I apologize, but I encountered an error generating a response. Please try again."
    
    async def generate_streaming_response(
        self,
        messages: List[ChatMessage],
        context: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response for real-time chat."""
        # Prepare messages
        api_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context}
        ]
        
        for msg in messages:
            api_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        # Calculate token budget
        prompt_tokens = sum(
            len(self.encoding.encode(msg["content"]))
            for msg in api_messages
        )
        
        if max_tokens is None:
            max_tokens = min(
                4000,
                settings.max_context_length - prompt_tokens - 100
            )
        
        # Stream response
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(
                "Failed to generate streaming response",
                error=str(e)
            )
            yield "I apologize, but I encountered an error. Please try again."
    
    async def generate_summary(
        self,
        text: str,
        max_length: int = 500
    ) -> str:
        """Generate a summary of the provided text."""
        prompt = f"""Please provide a concise summary of the following meeting content in {max_length} words or less:

{text}

Summary:"""
        
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        
        return await self.generate_response(
            messages=messages,
            context="",
            temperature=0.5,
            use_cache=True
        )
    
    async def extract_action_items(
        self,
        text: str
    ) -> List[Dict[str, Any]]:
        """Extract action items from meeting text."""
        prompt = """Please extract all action items from the following meeting content. 
For each action item, provide:
- description: What needs to be done
- assignee: Who is responsible (if mentioned)
- deadline: When it should be completed (if mentioned)
- priority: High/Medium/Low (based on context)

Format the response as a JSON array.

Meeting content:
""" + text
        
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        
        response = await self.generate_response(
            messages=messages,
            context="",
            temperature=0.3,  # Lower temperature for structured output
            use_cache=False
        )
        
        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response
            
            action_items = json.loads(json_str)
            
            if isinstance(action_items, list):
                return action_items
            else:
                return []
                
        except Exception as e:
            logger.error(
                "Failed to parse action items JSON",
                error=str(e),
                response_preview=response[:200]
            )
            return []
    
    async def answer_question(
        self,
        question: str,
        context: str,
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> str:
        """Answer a specific question based on context."""
        # Build focused prompt
        prompt = f"""Based on the meeting context provided, please answer the following question:

Question: {question}

Please provide a clear and specific answer based only on the information available in the context. If the context doesn't contain enough information to answer the question, please say so."""
        
        messages = conversation_history or []
        messages.append(ChatMessage(role=MessageRole.USER, content=prompt))
        
        return await self.generate_response(
            messages=messages,
            context=context,
            temperature=0.5,
            use_cache=True
        )
    
    async def classify_query_intent(
        self,
        query: str
    ) -> Dict[str, Any]:
        """Classify the intent of a user query."""
        prompt = f"""Analyze the following user query and classify its intent.

Query: "{query}"

Possible intents:
- summary: User wants a summary of meetings or discussions
- action_items: User is looking for action items or tasks
- search: User is searching for specific information
- participants: User wants information about meeting participants
- timeline: User is asking about dates, schedules, or timelines
- clarification: User needs clarification on previous responses
- general: General question or discussion

Respond with:
1. Primary intent (one of the above)
2. Confidence score (0-1)
3. Key entities mentioned (people, dates, topics)

Format as JSON."""
        
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        
        response = await self.generate_response(
            messages=messages,
            context="",
            temperature=0.3,
            use_cache=True
        )
        
        try:
            # Parse JSON response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response
            
            return json.loads(json_str)
            
        except Exception as e:
            logger.error(
                "Failed to parse intent classification",
                error=str(e)
            )
            
            return {
                "intent": "general",
                "confidence": 0.5,
                "entities": []
            }
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(self.encoding.encode(text))
    
    async def clear_cache(self) -> None:
        """Clear the response cache."""
        await self.cache.clear()
        logger.info("Cleared LLM response cache")


# Global instance
llm_service = LLMService()
