"""
Context generation module.
Creates contextual summaries for document chunks using LLM.
"""

import logging
from typing import List, Optional, Dict, Tuple
import asyncio
from dataclasses import dataclass

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class ChunkWithContext:
    """Represents a chunk with its contextual summary"""
    original_chunk: str
    context: str
    combined_content: str
    chunk_index: int
    token_usage: Optional[Dict[str, int]] = None

class ContextGenerator:
    """Generates contextual summaries for document chunks"""
    
    def __init__(self):
        """Initialize context generator with OpenAI client"""
        self.client = openai.OpenAI(api_key=settings.openai.api_key)
        self.model = settings.openai.context_model
        
        # Prompt template
        self.context_prompt = """<document> 
{document} 
</document> 
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk} 
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate_context(self, chunk: str, full_document: str, 
                        chunk_index: int) -> Tuple[ChunkWithContext, Dict[str, int]]:
        """Generate contextual summary for a single chunk
        
        Args:
            chunk: The chunk content
            full_document: The full document text
            chunk_index: Index of this chunk
            
        Returns:
            Tuple of (ChunkWithContext object, token usage dict)
        """
        try:
            # Prepare prompt
            prompt = self.context_prompt.format(
                document=full_document[:8000],  # Limit document size for context
                chunk=chunk
            )
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a legal document analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            context = response.choices[0].message.content.strip()
            
            # Extract token usage
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            # Combine context and chunk
            combined = f"{context}\n---\n{chunk}"
            
            return ChunkWithContext(
                original_chunk=chunk,
                context=context,
                combined_content=combined,
                chunk_index=chunk_index,
                token_usage=token_usage
            ), token_usage
            
        except Exception as e:
            logger.error(f"Error generating context for chunk {chunk_index}: {str(e)}")
            # Return chunk without context on error
            return ChunkWithContext(
                original_chunk=chunk,
                context="",
                combined_content=chunk,
                chunk_index=chunk_index,
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            ), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    async def generate_contexts_batch(self, chunks: List[str], 
                                    full_document: str) -> Tuple[List[ChunkWithContext], Dict[str, int]]:
        """Generate contexts for multiple chunks in parallel
        
        Args:
            chunks: List of chunk contents
            full_document: The full document text
            
        Returns:
            Tuple of (List of ChunkWithContext objects, total token usage)
        """
        logger.info(f"Generating contexts for {len(chunks)} chunks")
        
        # Track total token usage
        total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
        # Create tasks for parallel processing
        tasks = []
        for i, chunk in enumerate(chunks):
            task = asyncio.create_task(
                self._async_generate_context(chunk, full_document, i)
            )
            tasks.append(task)
        
        # Process in batches to avoid rate limits
        batch_size = 5
        results = []
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch)
            
            # Process results and accumulate token usage
            for chunk_with_context, usage in batch_results:
                results.append(chunk_with_context)
                total_usage["prompt_tokens"] += usage["prompt_tokens"]
                total_usage["completion_tokens"] += usage["completion_tokens"]
                total_usage["total_tokens"] += usage["total_tokens"]
            
            # Small delay between batches
            if i + batch_size < len(tasks):
                await asyncio.sleep(1)
        
        logger.info(f"Generated contexts for {len(results)} chunks, total tokens: {total_usage['total_tokens']}")
        return results, total_usage
    
    async def _async_generate_context(self, chunk: str, full_document: str, 
                                    chunk_index: int) -> Tuple[ChunkWithContext, Dict[str, int]]:
        """Async wrapper for context generation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.generate_context, 
            chunk, 
            full_document, 
            chunk_index
        )
    
    def generate_contexts_sync(self, chunks: List[str], 
                             full_document: str) -> Tuple[List[ChunkWithContext], Dict[str, int]]:
        """Synchronous wrapper for context generation
        
        Args:
            chunks: List of chunk contents
            full_document: The full document text
            
        Returns:
            Tuple of (List of ChunkWithContext objects, total token usage)
        """
        # Run async function in new event loop
        return asyncio.run(self.generate_contexts_batch(chunks, full_document))
    
    def validate_context(self, context: str) -> bool:
        """Validate that generated context is appropriate
        
        Args:
            context: Generated context string
            
        Returns:
            True if context is valid
        """
        if not context or len(context.strip()) < 10:
            return False
        
        # Check for common LLM refusal patterns
        refusal_patterns = [
            "I cannot", "I don't", "I'm not able", 
            "As an AI", "I apologize"
        ]
        
        context_lower = context.lower()
        for pattern in refusal_patterns:
            if pattern.lower() in context_lower:
                return False
        
        return True