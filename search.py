"""
Search Module
Implements similarity search with source citations and LLM answer generation.
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from embeddings import EmbeddingModel, get_embedding_model, DEFAULT_MODEL
from vector_store import VectorStore
from llm_integration import LLMIntegration, create_llm_integration


@dataclass
class SearchResult:
    """Represents a search result with source citation."""
    text: str
    source_file: str
    score: float
    chunk_index: int
    total_chunks: int

    def format_citation(self) -> str:
        """Format the source citation."""
        return f"[{self.source_file} - chunk {self.chunk_index + 1}/{self.total_chunks}]"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'source_file': self.source_file,
            'score': self.score,
            'chunk_index': self.chunk_index,
            'total_chunks': self.total_chunks,
            'citation': self.format_citation()
        }


@dataclass
class SearchResponse:
    """Complete search response with optional LLM answer."""
    chunks: List[SearchResult]
    answer: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    tokens_used: Optional[int] = None
    llm_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'chunks': [c.to_dict() for c in self.chunks],
            'answer': self.answer,
            'llm_provider': self.llm_provider,
            'llm_model': self.llm_model,
            'tokens_used': self.tokens_used,
            'llm_error': self.llm_error
        }


class DocumentSearch:
    """
    Main search class that combines embedding model and vector store.
    Supports LLM answer generation.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        llm_integration: Optional[LLMIntegration] = None,
        model_name: str = DEFAULT_MODEL
    ):
        """
        Initialize the search system.

        Args:
            vector_store: Pre-loaded vector store
            embedding_model: Pre-loaded embedding model
            llm_integration: Optional LLM integration for answer generation
            model_name: Model name to use if embedding_model not provided
        """
        self.embedding_model = embedding_model or get_embedding_model(model_name)
        self.vector_store = vector_store
        self.llm_integration = llm_integration

        # Ensure model is loaded
        if self.embedding_model is not None:
            self.embedding_model.load()

    def load_index(self, index_path: str):
        """
        Load a saved vector store index.

        Args:
            index_path: Path to the saved index directory
        """
        self.vector_store = VectorStore(index_path=index_path)
        print(f"Loaded index from {index_path}")
        print(f"  - Total vectors: {self.vector_store.index.ntotal}")

    def set_llm(self, llm_integration: LLMIntegration):
        """
        Set or update the LLM integration.

        Args:
            llm_integration: LLMIntegration instance
        """
        self.llm_integration = llm_integration

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search for relevant documents.

        Args:
            query: Search query string
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1) to include result

        Returns:
            List of SearchResult objects
        """
        if self.vector_store is None or self.vector_store.is_empty():
            print("Warning: No index loaded or index is empty")
            return []

        # Encode query
        query_embedding = self.embedding_model.encode(query)

        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k * 2)

        # Convert to SearchResult objects
        search_results = []
        for result in results:
            metadata = result['metadata']

            # Apply score threshold if specified
            if score_threshold is not None and result['score'] < score_threshold:
                continue

            search_result = SearchResult(
                text=metadata.get('text', ''),
                source_file=metadata.get('source_file', 'Unknown'),
                score=result['score'],
                chunk_index=metadata.get('chunk_index', 0),
                total_chunks=metadata.get('total_chunks', 1)
            )
            search_results.append(search_result)

        # Sort by score and limit to top_k
        search_results.sort(key=lambda x: x.score, reverse=True)
        return search_results[:top_k]

    def search_with_answer(
        self,
        query: str,
        top_k: int = 5,
        generate_answer: bool = True,
        max_tokens: int = 500
    ) -> SearchResponse:
        """
        Search for documents and optionally generate an LLM answer.

        Args:
            query: Search query string
            top_k: Number of chunks to retrieve
            generate_answer: Whether to generate an LLM answer
            max_tokens: Maximum tokens for LLM answer

        Returns:
            SearchResponse with chunks and optional answer
        """
        # Get search results
        chunks = self.search(query, top_k=top_k)

        # Generate answer if requested and LLM is available
        answer = None
        llm_provider = None
        llm_model = None
        tokens_used = None
        llm_error = None

        if generate_answer and self.llm_integration and self.llm_integration.is_available():
            try:
                context_chunks = [c.to_dict() for c in chunks]
                llm_response = self.llm_integration.generate_answer(
                    query=query,
                    context_chunks=context_chunks,
                    max_tokens=max_tokens
                )

                answer = llm_response.answer
                llm_provider = llm_response.provider
                llm_model = llm_response.model
                tokens_used = llm_response.tokens_used
                llm_error = llm_response.error

            except Exception as e:
                llm_error = str(e)
                print(f"LLM answer generation failed: {e}")

        return SearchResponse(
            chunks=chunks,
            answer=answer,
            llm_provider=llm_provider,
            llm_model=llm_model,
            tokens_used=tokens_used,
            llm_error=llm_error
        )

    def search_with_context(
        self,
        query: str,
        top_k: int = 5,
        context_window: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Search with surrounding context from the same document.

        Args:
            query: Search query
            top_k: Number of primary results
            context_window: Number of adjacent chunks to include

        Returns:
            List of results with context
        """
        results = self.search(query, top_k=top_k)

        enriched_results = []
        for result in results:
            # Get adjacent chunks if available
            context_before = []
            context_after = []

            # This would require storing all chunks or having access to the original document
            # For now, we just return the result with its metadata

            enriched_result = {
                'result': result.to_dict(),
                'context_before': context_before,
                'context_after': context_after,
                'full_context': result.text
            }
            enriched_results.append(enriched_result)

        return enriched_results

    def format_results(
        self,
        results: List[SearchResult],
        include_scores: bool = True,
        max_text_length: int = 500
    ) -> str:
        """
        Format search results as a readable string.

        Args:
            results: List of SearchResult objects
            include_scores: Whether to include similarity scores
            max_text_length: Maximum length of text to display

        Returns:
            Formatted string
        """
        if not results:
            return "No results found."

        lines = []
        lines.append(f"Found {len(results)} relevant document(s):\n")

        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result.format_citation()}")
            if include_scores:
                lines.append(f"   Relevance: {result.score:.2%}")

            text = result.text
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."

            # Indent text
            indented_text = "\n   ".join(text.split('\n'))
            lines.append(f"   {indented_text}")
            lines.append("")

        return "\n".join(lines)

    def format_response(
        self,
        response: SearchResponse,
        include_scores: bool = True,
        max_text_length: int = 500
    ) -> str:
        """
        Format complete search response with answer.

        Args:
            response: SearchResponse object
            include_scores: Whether to include similarity scores
            max_text_length: Maximum length of text to display

        Returns:
            Formatted string
        """
        lines = []

        # Add LLM answer if available
        if response.answer:
            lines.append("=" * 60)
            lines.append("AI-GENERATED ANSWER")
            lines.append("=" * 60)
            lines.append(response.answer)
            lines.append("")
            if response.llm_provider:
                lines.append(f"(Generated by {response.llm_provider}/{response.llm_model})")
            if response.tokens_used:
                lines.append(f"Tokens used: {response.tokens_used}")
            lines.append("")

        # Add source chunks
        lines.append("=" * 60)
        lines.append("SOURCE CHUNKS")
        lines.append("=" * 60)
        lines.append(self.format_results(response.chunks, include_scores, max_text_length))

        return "\n".join(lines)

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded index."""
        if self.vector_store is None:
            return {'status': 'No index loaded'}

        return self.vector_store.get_stats()


def create_search_engine(
    index_path: str,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    api_key: Optional[str] = None
) -> DocumentSearch:
    """
    Create a search engine from a saved index with optional LLM.

    Args:
        index_path: Path to the saved index
        llm_provider: Optional LLM provider ('ollama' or 'openai')
        llm_model: Optional LLM model name
        api_key: Optional API key for OpenAI

    Returns:
        Configured DocumentSearch instance
    """
    search = DocumentSearch()
    search.load_index(index_path)

    # Configure LLM if provider specified
    if llm_provider:
        try:
            kwargs = {'provider': llm_provider}
            if llm_model:
                kwargs['model'] = llm_model
            if api_key:
                kwargs['api_key'] = api_key

            llm = create_llm_integration(**kwargs)
            search.set_llm(llm)
        except Exception as e:
            print(f"Warning: Could not initialize LLM: {e}")

    return search


def search_documents(
    query: str,
    index_path: str,
    top_k: int = 5,
    generate_answer: bool = False,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None
) -> SearchResponse:
    """
    Convenience function to search without managing instances.

    Args:
        query: Search query
        index_path: Path to index
        top_k: Number of results
        generate_answer: Whether to generate LLM answer
        llm_provider: Optional LLM provider
        llm_model: Optional LLM model

    Returns:
        SearchResponse with chunks and optional answer
    """
    search = create_search_engine(index_path, llm_provider, llm_model)
    return search.search_with_answer(query, top_k=top_k, generate_answer=generate_answer)


if __name__ == "__main__":
    # Test the search module
    print("Testing search module...\n")

    # This requires an existing index
    # For testing, we'll create a simple mock

    from vector_store import VectorStore
    from embeddings import EmbeddingModel

    # Create sample data
    model = EmbeddingModel()
    texts = [
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "FAISS is a library for efficient similarity search.",
        "Document retrieval is important for RAG systems.",
        "Embeddings convert text to numerical vectors."
    ]

    print("Creating sample embeddings...")
    embeddings = model.encode(texts)

    # Create vector store
    store = VectorStore(embedding_dim=model.embedding_dim)
    metadata = [
        {'text': t, 'source_file': f'doc_{i}.txt', 'chunk_index': i, 'total_chunks': len(texts)}
        for i, t in enumerate(texts)
    ]
    store.add_embeddings(embeddings, metadata)

    # Create search engine
    search = DocumentSearch(vector_store=store, embedding_model=model)

    # Test queries
    queries = [
        "What is Python?",
        "Tell me about machine learning",
        "How does similarity search work?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = search.search(query, top_k=2)
        print(search.format_results(results, max_text_length=100))

    # Test search with answer (without LLM)
    print("\n" + "=" * 60)
    print("Testing search with answer generation (no LLM)")
    print("=" * 60)
    response = search.search_with_answer("What is Python?", generate_answer=False)
    print(f"Found {len(response.chunks)} chunks")
    print(f"Answer generated: {response.answer is not None}")

    print("\nTest completed successfully!")
