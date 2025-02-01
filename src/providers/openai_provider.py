from typing import AsyncGenerator, List
from openai import AsyncOpenAI
from .base import LLMProvider, ProviderConfig

class OpenAIConfig(ProviderConfig):
    """OpenAI-specific configuration."""
    model: str = "gpt-4"
    embedding_model: str = "text-embedding-ada-002"

class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""
    
    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=config.api_key)
    
    async def generate_response(self, prompt: str) -> str:
        """Generate a response using OpenAI's API."""
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        return response.choices[0].message.content

    async def stream_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream a response using OpenAI's API."""
        stream = await self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding using OpenAI's API."""
        response = await self.client.embeddings.create(
            model=self.config.embedding_model,
            input=text
        )
        return response.data[0].embedding
