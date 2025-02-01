from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, Optional, List
from pydantic import BaseModel

class ProviderConfig(BaseModel):
    """Base configuration for LLM providers."""
    api_key: str
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    async def generate_response(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    async def stream_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream a response from the LLM."""
        pass

    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for the given text."""
        pass
