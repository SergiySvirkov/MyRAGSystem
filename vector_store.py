"""
Vector Store Module
Manages FAISS index with metadata for efficient similarity search.
Updated for multilingual embeddings (384-dim).
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import os
import shutil

from embeddings import DEFAULT_EMBEDDING_DIM


class VectorStore:
    """
    FAISS-based vector store with metadata management.
    """

    def __init__(self, embedding_dim: int = DEFAULT_EMBEDDING_DIM, index_path: Optional[str] = None):
        """
        Initialize the vector store.

        Args:
            embedding_dim: Dimension of embeddings (384 for multilingual-MiniLM-L12-v2)
            index_path: Path to load existing index from
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.index_path = index_path

        if index_path and os.path.exists(index_path):
            self.load(index_path)
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create a new empty FAISS index."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")

        # Create a flat L2 index (exact search)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        print(f"Created new FAISS index with dimension {self.embedding_dim}")

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata_list: List[Dict[str, Any]]
    ):
        """
        Add embeddings and their metadata to the index.

        Args:
            embeddings: Numpy array of shape [n, embedding_dim]
            metadata_list: List of metadata dictionaries for each embedding
        """
        if embeddings.shape[0] != len(metadata_list):
            raise ValueError(
                f"Number of embeddings ({embeddings.shape[0]}) "
                f"must match number of metadata entries ({len(metadata_list)})"
            )

        # Ensure embeddings are float32
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Add to FAISS index
        self.index.add(embeddings)

        # Store metadata
        self.metadata.extend(metadata_list)

        print(f"Added {len(metadata_list)} embeddings to index. Total: {len(self.metadata)}")

    def add_chunks(self, chunks: List, embeddings: np.ndarray):
        """
        Add chunks with their embeddings to the index.

        Args:
            chunks: List of Chunk objects
            embeddings: Corresponding embeddings
        """
        metadata_list = []
        for chunk in chunks:
            if hasattr(chunk, 'to_dict'):
                metadata_list.append(chunk.to_dict())
            elif isinstance(chunk, dict):
                metadata_list.append(chunk)
            else:
                metadata_list.append({'text': str(chunk)})

        self.add_embeddings(embeddings, metadata_list)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of results with metadata and distances
        """
        if self.index.ntotal == 0:
            print("Warning: Index is empty")
            return []

        # Ensure query is float32 and 2D
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                result = {
                    'index': int(idx),
                    'distance': float(dist),
                    'score': float(1 / (1 + dist)),  # Convert distance to similarity score
                    'metadata': self.metadata[idx]
                }
                results.append(result)

        return results

    def save(self, path: str):
        """
        Save the index and metadata to disk.

        Args:
            path: Directory path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_file = path / "index.faiss"
        try:
            import faiss
            faiss.write_index(self.index, str(index_file))
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
            raise

        # Save metadata
        metadata_file = path / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)

        # Save config
        config = {
            'embedding_dim': self.embedding_dim,
            'num_vectors': len(self.metadata),
            'model_name': 'paraphrase-multilingual-MiniLM-L12-v2'
        }
        config_file = path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Saved index to {path}")
        print(f"  - Vectors: {len(self.metadata)}")
        print(f"  - Dimension: {self.embedding_dim}")

    def load(self, path: str):
        """
        Load the index and metadata from disk.

        Args:
            path: Directory path to load from
        """
        path = Path(path)

        # Load config
        config_file = path / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            self.embedding_dim = config.get('embedding_dim', DEFAULT_EMBEDDING_DIM)

        # Load FAISS index
        index_file = path / "index.faiss"
        if index_file.exists():
            try:
                import faiss
                self.index = faiss.read_index(str(index_file))
                print(f"Loaded FAISS index from {index_file}")
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                raise
        else:
            raise FileNotFoundError(f"Index file not found: {index_file}")

        # Load metadata
        metadata_file = path / "metadata.pkl"
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"Loaded {len(self.metadata)} metadata entries")
        else:
            print(f"Warning: Metadata file not found: {metadata_file}")
            self.metadata = []

    def clear(self):
        """Clear the index and metadata."""
        self.metadata = []
        self._create_new_index()
        print("Index cleared")

    def is_empty(self) -> bool:
        """Check if the index is empty."""
        return self.index is None or self.index.ntotal == 0

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        return {
            'total_vectors': len(self.metadata),
            'embedding_dim': self.embedding_dim,
            'is_empty': self.is_empty()
        }

    def get_all_embeddings(self) -> np.ndarray:
        """
        Get all embeddings from the index.

        Returns:
            Numpy array of all embeddings
        """
        if self.is_empty():
            return np.array([])

        # Reconstruct embeddings from FAISS index
        return self.index.reconstruct_n(0, self.index.ntotal)

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """Get all metadata entries."""
        return self.metadata.copy()


def create_vector_store_from_chunks(
    chunks: List,
    embedding_model,
    index_path: Optional[str] = None
) -> VectorStore:
    """
    Create a vector store from chunks.

    Args:
        chunks: List of chunks
        embedding_model: Embedding model instance
        index_path: Optional path to save index

    Returns:
        Populated VectorStore
    """
    from embeddings import get_embedding_model

    if embedding_model is None:
        embedding_model = get_embedding_model()

    # Get embedding dimension
    embedding_dim = embedding_model.embedding_dim

    # Create vector store
    store = VectorStore(embedding_dim=embedding_dim)

    # Encode chunks
    print(f"Encoding {len(chunks)} chunks...")
    embeddings = embedding_model.encode_chunks(chunks)

    # Add to store
    store.add_chunks(chunks, embeddings)

    # Save if path provided
    if index_path:
        store.save(index_path)

    return store


def reindex_documents(
    chunks: List,
    embedding_model,
    index_path: str,
    backup_old: bool = True
) -> VectorStore:
    """
    Re-index documents with a new embedding model.

    Args:
        chunks: List of chunks to index
        embedding_model: New embedding model instance
        index_path: Path to save the new index
        backup_old: Whether to backup the old index

    Returns:
        New VectorStore with re-indexed embeddings
    """
    path = Path(index_path)

    # Backup old index if it exists
    if backup_old and path.exists():
        backup_path = Path(str(path) + "_backup")
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.copytree(path, backup_path)
        print(f"Backed up old index to {backup_path}")

    # Clear existing index
    if path.exists():
        shutil.rmtree(path)
        print(f"Cleared old index at {path}")

    # Create new vector store with updated embeddings
    store = create_vector_store_from_chunks(chunks, embedding_model, index_path)
    print(f"Re-indexing complete. New index has {store.index.ntotal} vectors.")

    return store


if __name__ == "__main__":
    # Test the vector store
    print("Testing vector store...\n")

    # Create sample embeddings
    dim = DEFAULT_EMBEDDING_DIM
    n_vectors = 10
    embeddings = np.random.randn(n_vectors, dim).astype(np.float32)

    # Normalize embeddings (common practice)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Create metadata
    metadata = [
        {'text': f'Sample text {i}', 'source': f'doc_{i}.txt'}
        for i in range(n_vectors)
    ]

    # Create and populate store
    store = VectorStore(embedding_dim=dim)
    store.add_embeddings(embeddings, metadata)

    print(f"\nIndex stats: {store.get_stats()}")

    # Test search
    query = np.random.randn(dim).astype(np.float32)
    query = query / np.linalg.norm(query)

    results = store.search(query, top_k=3)
    print(f"\nSearch results:")
    for r in results:
        print(f"  - Score: {r['score']:.4f}, Text: {r['metadata']['text']}")

    # Test save/load
    test_path = "/tmp/test_vector_store"
    store.save(test_path)

    new_store = VectorStore(index_path=test_path)
    print(f"\nLoaded index stats: {new_store.get_stats()}")

    # Test get_all_embeddings
    all_embeddings = new_store.get_all_embeddings()
    print(f"Retrieved all embeddings shape: {all_embeddings.shape}")

    # Cleanup
    import shutil
    shutil.rmtree(test_path, ignore_errors=True)
    print("\nTest completed successfully!")
