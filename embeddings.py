"""
Embeddings Module
Loads and uses sentence-transformers model for creating embeddings.
Supports multilingual embeddings with paraphrase-multilingual-MiniLM-L12-v2.
"""

import numpy as np
from typing import List, Union
import os

# Suppress transformers warnings
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# Default multilingual model
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_DIM = 384


class EmbeddingModel:
    """
    Wrapper for sentence-transformers embedding model.
    Supports multilingual embeddings.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model name for sentence-transformers
        """
        self.model_name = model_name
        self.model = None
        self._embedding_dim = None

    def load(self):
        """
        Load the embedding model.
        This is called automatically on first use if not already loaded.
        """
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                self._embedding_dim = self.model.get_sentence_embedding_dimension()
                print(f"Model loaded. Embedding dimension: {self._embedding_dim}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load model {self.model_name}: {e}")

        return self

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self._embedding_dim is None:
            self.load()
        return self._embedding_dim

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        self.load()

        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        # Filter out empty texts
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)

        if not valid_texts:
            # Return zero embeddings for all empty inputs
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)

        # Encode valid texts
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        # If there were empty texts, reconstruct full array
        if len(valid_texts) != len(texts):
            full_embeddings = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
            for idx, valid_idx in enumerate(valid_indices):
                full_embeddings[valid_idx] = embeddings[idx]
            embeddings = full_embeddings

        return embeddings[0] if single_input else embeddings

    def encode_chunks(self, chunks: List, batch_size: int = 32) -> np.ndarray:
        """
        Encode chunks from the chunker module.

        Args:
            chunks: List of Chunk objects or dictionaries with 'text' key
            batch_size: Batch size for encoding

        Returns:
            Numpy array of embeddings
        """
        texts = []
        for chunk in chunks:
            if hasattr(chunk, 'text'):
                texts.append(chunk.text)
            elif isinstance(chunk, dict):
                texts.append(chunk.get('text', ''))
            else:
                texts.append(str(chunk))

        return self.encode(texts, batch_size=batch_size)


# Global model instance for reuse
_model_instance = None


def get_embedding_model(model_name: str = DEFAULT_MODEL) -> EmbeddingModel:
    """
    Get or create a singleton embedding model instance.

    Args:
        model_name: Model name to use

    Returns:
        EmbeddingModel instance
    """
    global _model_instance
    if _model_instance is None or _model_instance.model_name != model_name:
        _model_instance = EmbeddingModel(model_name)
    return _model_instance


def encode_texts(
    texts: Union[str, List[str]],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32
) -> np.ndarray:
    """
    Convenience function to encode texts without managing model instance.

    Args:
        texts: Texts to encode
        model_name: Model name to use
        batch_size: Batch size for encoding

    Returns:
        Numpy array of embeddings
    """
    model = get_embedding_model(model_name)
    return model.encode(texts, batch_size=batch_size)


if __name__ == "__main__":
    # Test the embeddings module
    print("Testing embeddings module...\n")

    # Test single text
    model = EmbeddingModel()
    text = "This is a test sentence."
    embedding = model.encode(text)
    print(f"Single text embedding shape: {embedding.shape}")
    print(f"Embedding dimension: {model.embedding_dim}")
    print(f"Sample values: {embedding[:5]}")
    print()

    # Test multiple texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language."
    ]
    embeddings = model.encode(texts)
    print(f"Multiple texts embedding shape: {embeddings.shape}")
    print(f"Embeddings computed successfully!")
    print()

    # Test multilingual text
    print("Testing multilingual support...")
    multilingual_texts = [
        "This is English text.",
        "Ceci est un texte français.",
        "Dies ist ein deutscher Text.",
        "Este es un texto en español.",
        "这是中文文本。"
    ]
    ml_embeddings = model.encode(multilingual_texts)
    print(f"Multilingual texts embedding shape: {ml_embeddings.shape}")
    print("Multilingual embeddings computed successfully!")
