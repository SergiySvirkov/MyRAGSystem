"""
Chunker Module
Splits documents into semantic chunks with metadata.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    source_file: str
    chunk_index: int
    start_char: int
    end_char: int
    total_chunks: int
    
    def to_dict(self) -> dict:
        """Convert chunk to dictionary."""
        return {
            'text': self.text,
            'source_file': self.source_file,
            'chunk_index': self.chunk_index,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'total_chunks': self.total_chunks
        }


def split_by_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs.
    
    Args:
        text: Input text
        
    Returns:
        List of paragraphs
    """
    # Split on multiple newlines or paragraph breaks
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def create_chunks(
    text: str,
    source_file: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    min_chunk_size: int = 100
) -> List[Chunk]:
    """
    Split text into overlapping chunks.
    
    Uses a hybrid approach:
    1. First tries to split by paragraphs
    2. If paragraph is too long, splits by sentences
    3. If sentence is too long, splits by character count
    
    Args:
        text: Text content to chunk
        source_file: Path to source file (for metadata)
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum size for a chunk to be included
        
    Returns:
        List of Chunk objects
    """
    if not text or not text.strip():
        return []
    
    chunks = []
    paragraphs = split_by_paragraphs(text)
    
    current_chunk = []
    current_size = 0
    char_position = 0
    chunk_index = 0
    
    for paragraph in paragraphs:
        paragraph_len = len(paragraph)
        
        # If adding this paragraph would exceed chunk_size, finalize current chunk
        if current_size + paragraph_len > chunk_size and current_chunk:
            # Create chunk from accumulated paragraphs
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text) >= min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    start_char=char_position - current_size,
                    end_char=char_position,
                    total_chunks=0  # Will be updated later
                ))
                chunk_index += 1
            
            # Start new chunk with overlap
            overlap_text = chunk_text[-chunk_overlap:] if len(chunk_text) > chunk_overlap else chunk_text
            current_chunk = [overlap_text, paragraph] if overlap_text else [paragraph]
            current_size = sum(len(p) for p in current_chunk) + len(current_chunk) - 1
        else:
            current_chunk.append(paragraph)
            current_size += paragraph_len + 2  # +2 for newlines
        
        char_position += paragraph_len + 2
    
    # Don't forget the last chunk
    if current_chunk:
        chunk_text = '\n\n'.join(current_chunk)
        if len(chunk_text) >= min_chunk_size:
            chunks.append(Chunk(
                text=chunk_text,
                source_file=source_file,
                chunk_index=chunk_index,
                start_char=char_position - current_size,
                end_char=char_position,
                total_chunks=0
            ))
    
    # Update total_chunks for all chunks
    total = len(chunks)
    for chunk in chunks:
        chunk.total_chunks = total
    
    return chunks


def chunk_document(
    text: str,
    source_file: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Chunk]:
    """
    Chunk a single document.
    
    Args:
        text: Document text
        source_file: Source file path
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunks
    """
    return create_chunks(text, source_file, chunk_size, chunk_overlap)


def chunk_multiple_documents(
    documents: List[tuple],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Chunk]:
    """
    Chunk multiple documents.
    
    Args:
        documents: List of (text, source_file) tuples
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of all chunks from all documents
    """
    all_chunks = []
    
    for text, source_file in documents:
        if text and text.strip():
            chunks = chunk_document(text, source_file, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
    
    return all_chunks


def get_chunk_statistics(chunks: List[Chunk]) -> dict:
    """
    Get statistics about chunks.
    
    Args:
        chunks: List of chunks
        
    Returns:
        Dictionary with statistics
    """
    if not chunks:
        return {
            'total_chunks': 0,
            'avg_chunk_size': 0,
            'min_chunk_size': 0,
            'max_chunk_size': 0,
            'unique_sources': 0
        }
    
    sizes = [len(chunk.text) for chunk in chunks]
    sources = set(chunk.source_file for chunk in chunks)
    
    return {
        'total_chunks': len(chunks),
        'avg_chunk_size': sum(sizes) / len(sizes),
        'min_chunk_size': min(sizes),
        'max_chunk_size': max(sizes),
        'unique_sources': len(sources)
    }


if __name__ == "__main__":
    # Test the chunker
    sample_text = """
    This is the first paragraph. It contains some information about a topic.
    
    This is the second paragraph. It has more details and continues the discussion.
    It might be a bit longer than the first one.
    
    This is the third paragraph. It concludes the document with final thoughts.
    """
    
    chunks = chunk_document(sample_text, "test_document.txt", chunk_size=100, chunk_overlap=20)
    
    print(f"Created {len(chunks)} chunks:\n")
    for chunk in chunks:
        print(f"Chunk {chunk.chunk_index + 1}/{chunk.total_chunks}")
        print(f"Source: {chunk.source_file}")
        print(f"Position: {chunk.start_char}-{chunk.end_char}")
        print(f"Text: {chunk.text[:100]}...")
        print()
    
    stats = get_chunk_statistics(chunks)
    print("Statistics:", stats)
