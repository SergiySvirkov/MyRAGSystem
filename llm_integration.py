"""
LLM Integration Module
Provides unified interface for Ollama (local) and OpenAI API.
Generates natural language answers from retrieved context.
"""

import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import json


@dataclass
class LLMResponse:
    """Response from LLM."""
    answer: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    error: Optional[str] = None


class LLMProvider:
    """Base class for LLM providers."""

    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        max_tokens: int = 500
    ) -> LLMResponse:
        """
        Generate an answer based on retrieved context.

        Args:
            query: User query
            context_chunks: Retrieved context chunks with metadata
            max_tokens: Maximum tokens for response

        Returns:
            LLMResponse with answer and metadata
        """
        raise NotImplementedError

    def is_available(self) -> bool:
        """Check if the provider is available/configured."""
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    DEFAULT_MODEL = "llama3.2"
    DEFAULT_HOST = "http://localhost:11434"

    def __init__(self, model: str = DEFAULT_MODEL, host: Optional[str] = None):
        """
        Initialize Ollama provider.

        Args:
            model: Ollama model name
            host: Ollama host URL
        """
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", self.DEFAULT_HOST)
        self._client = None

    def _get_client(self):
        """Get or create Ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.host)
            except ImportError:
                raise ImportError("ollama package required. Install with: pip install ollama")
        return self._client

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            client = self._get_client()
            # Try to list models to check connectivity
            client.list()
            return True
        except Exception as e:
            print(f"Ollama not available: {e}")
            return False

    def _build_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Build prompt with context."""
        context_text = "\n\n".join([
            f"[Source: {chunk.get('source_file', 'Unknown')}]\n{chunk.get('text', '')}"
            for chunk in context_chunks
        ])

        prompt = f"""You are a helpful assistant answering questions based on the provided document context.

Context from documents:
{context_text}

Question: {query}

Please provide a clear, concise answer based on the context above. If the context doesn't contain enough information to answer the question, say so.

Answer:"""
        return prompt

    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        max_tokens: int = 500
    ) -> LLMResponse:
        """Generate answer using Ollama."""
        try:
            client = self._get_client()
            prompt = self._build_prompt(query, context_chunks)

            response = client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'num_predict': max_tokens,
                    'temperature': 0.7
                }
            )

            return LLMResponse(
                answer=response['response'].strip(),
                provider="ollama",
                model=self.model,
                tokens_used=response.get('eval_count', None)
            )

        except Exception as e:
            return LLMResponse(
                answer="",
                provider="ollama",
                model=self.model,
                error=str(e)
            )

    def list_models(self) -> List[str]:
        """List available Ollama models."""
        try:
            client = self._get_client()
            models = client.list()
            return [m['name'] for m in models.get('models', [])]
        except Exception as e:
            print(f"Error listing Ollama models: {e}")
            return []


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    DEFAULT_MODEL = "gpt-3.5-turbo"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        """
        Initialize OpenAI provider.

        Args:
            model: OpenAI model name
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                if not self.api_key:
                    raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        return self._client

    def is_available(self) -> bool:
        """Check if OpenAI API key is configured."""
        return self.api_key is not None and len(self.api_key) > 0

    def _build_messages(self, query: str, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Build messages for chat completion."""
        context_text = "\n\n".join([
            f"[Source: {chunk.get('source_file', 'Unknown')}]\n{chunk.get('text', '')}"
            for chunk in context_chunks
        ])

        system_prompt = """You are a helpful assistant answering questions based on the provided document context. Provide clear, concise answers based on the context. If the context doesn't contain enough information, say so."""

        user_prompt = f"""Context from documents:
{context_text}

Question: {query}

Please provide a clear, concise answer based on the context above."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        max_tokens: int = 500
    ) -> LLMResponse:
        """Generate answer using OpenAI API."""
        try:
            client = self._get_client()
            messages = self._build_messages(query, context_chunks)

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )

            return LLMResponse(
                answer=response.choices[0].message.content.strip(),
                provider="openai",
                model=self.model,
                tokens_used=response.usage.total_tokens if response.usage else None
            )

        except Exception as e:
            return LLMResponse(
                answer="",
                provider="openai",
                model=self.model,
                error=str(e)
            )


class LLMIntegration:
    """
    Unified LLM integration supporting multiple providers.
    """

    PROVIDERS = {
        'ollama': OllamaProvider,
        'openai': OpenAIProvider
    }

    def __init__(self, provider: str = "ollama", **kwargs):
        """
        Initialize LLM integration.

        Args:
            provider: Provider name ('ollama' or 'openai')
            **kwargs: Provider-specific arguments
        """
        self.provider_name = provider.lower()

        if self.provider_name not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(self.PROVIDERS.keys())}")

        provider_class = self.PROVIDERS[self.provider_name]
        self.provider = provider_class(**kwargs)

    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        max_tokens: int = 500
    ) -> LLMResponse:
        """
        Generate answer from context.

        Args:
            query: User query
            context_chunks: Retrieved context chunks
            max_tokens: Maximum tokens for response

        Returns:
            LLMResponse with answer
        """
        return self.provider.generate_answer(query, context_chunks, max_tokens)

    def is_available(self) -> bool:
        """Check if the configured provider is available."""
        return self.provider.is_available()

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider."""
        return {
            'provider': self.provider_name,
            'model': self.provider.model,
            'available': self.is_available()
        }


def create_llm_integration(
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    host: Optional[str] = None
) -> LLMIntegration:
    """
    Factory function to create LLM integration.

    Args:
        provider: 'ollama' or 'openai'
        model: Model name (provider-specific)
        api_key: API key for OpenAI
        host: Host URL for Ollama

    Returns:
        Configured LLMIntegration instance
    """
    kwargs = {}

    if model:
        kwargs['model'] = model
    if api_key:
        kwargs['api_key'] = api_key
    if host:
        kwargs['host'] = host

    return LLMIntegration(provider=provider, **kwargs)


if __name__ == "__main__":
    # Test LLM integration
    print("Testing LLM Integration...\n")

    # Sample context chunks
    sample_chunks = [
        {
            'text': 'Python is a high-level programming language known for its simplicity.',
            'source_file': 'python_guide.pdf',
            'chunk_index': 0
        },
        {
            'text': 'Machine learning is a subset of artificial intelligence that enables systems to learn from data.',
            'source_file': 'ml_intro.md',
            'chunk_index': 1
        }
    ]

    query = "What is Python?"

    # Test Ollama (if available)
    print("Testing Ollama provider...")
    try:
        ollama_llm = create_llm_integration("ollama")
        if ollama_llm.is_available():
            print("Ollama is available!")
            response = ollama_llm.generate_answer(query, sample_chunks, max_tokens=100)
            print(f"Answer: {response.answer[:200]}...")
            print(f"Tokens used: {response.tokens_used}")
        else:
            print("Ollama not available (server may not be running)")
    except Exception as e:
        print(f"Ollama test error: {e}")

    print()

    # Test OpenAI (if API key available)
    print("Testing OpenAI provider...")
    try:
        openai_llm = create_llm_integration("openai")
        if openai_llm.is_available():
            print("OpenAI is available!")
            response = openai_llm.generate_answer(query, sample_chunks, max_tokens=100)
            print(f"Answer: {response.answer[:200]}...")
            print(f"Tokens used: {response.tokens_used}")
        else:
            print("OpenAI not available (API key not set)")
    except Exception as e:
        print(f"OpenAI test error: {e}")

    print("\nLLM Integration test complete!")
