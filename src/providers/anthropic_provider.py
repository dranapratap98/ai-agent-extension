from typing import AsyncGenerator, List
from anthropic import AsyncAnthropic
from .base import LLMProvider, ProviderConfig

class AnthropicConfig(ProviderConfig):
    """Anthropic-specific configuration."""
    model: str = "claude-3-opus-20240229"

class AnthropicProvider(LLMProvider):
    """Anthropic API provider implementation."""
    
    def __init__(self, config: AnthropicConfig):
        super().__init__(config)
        self.client = AsyncAnthropic(api_key=config.api_key)
    
    async def generate_response(self, prompt: str) -> str:
        """Generate a response using Anthropic's API."""
        response = await self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content

    async def stream_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream a response using Anthropic's API."""
        stream = await self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        async for chunk in stream:
            if chunk.delta.text:
                yield chunk.delta.text

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding (Not currently supported by Anthropic)."""
        raise NotImplementedError("Anthropic does not currently support embeddings")
