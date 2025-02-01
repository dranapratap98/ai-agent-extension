import asyncio
from providers.openai_provider import OpenAIConfig, OpenAIProvider
from providers.anthropic_provider import AnthropicConfig, AnthropicProvider

# OpenAI Setup
openai_config = OpenAIConfig(api_key="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")
openai_provider = OpenAIProvider(openai_config)

# Anthropic Setup
anthropic_config = AnthropicConfig(api_key="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")
anthropic_provider = AnthropicProvider(anthropic_config)

async def main():
    prompt = "What is the capital of France?"
    
    # OpenAI Response
    openai_response = await openai_provider.generate_response(prompt)
    print("OpenAI Response:", openai_response)

    # Anthropic Response
    anthropic_response = await anthropic_provider.generate_response(prompt)
    print("Anthropic Response:", anthropic_response)

asyncio.run(main())
