"""
Document Scanner Module
Scans the workspace for PDF and Markdown files.
"""

import os
from pathlib import Path
from typing import List


def find_documents(root_dir: str = ".", recursive: bool = True) -> List[Path]:
    """
    Find all PDF and Markdown files in the specified directory.
    
    Args:
        root_dir: Root directory to search from
        recursive: Whether to search recursively
        
    Returns:
        List of Path objects for found documents
    """
    root_path = Path(root_dir).resolve()
    documents = []
    
    # Supported file extensions
    extensions = {'.pdf', '.md', '.markdown'}
    
    if recursive:
        # Walk through all subdirectories
        for ext in extensions:
            documents.extend(root_path.rglob(f'*{ext}'))
    else:
        # Only search in root directory
        for ext in extensions:
            documents.extend(root_path.glob(f'*{ext}'))
    
    # Sort for consistent ordering
    documents = sorted(documents)
    
    return documents


def get_document_info(doc_path: Path) -> dict:
    """
    Get metadata about a document.
    
    Args:
        doc_path: Path to the document
        
    Returns:
        Dictionary with document metadata
    """
    stat = doc_path.stat()
    return {
        'path': str(doc_path),
        'name': doc_path.name,
        'extension': doc_path.suffix.lower(),
        'size_bytes': stat.st_size,
        'modified_time': stat.st_mtime,
        'relative_path': str(doc_path.relative_to(Path.cwd()))
    }


def scan_workspace(root_dir: str = ".") -> List[dict]:
    """
    Scan workspace and return document info for all found documents.
    
    Args:
        root_dir: Root directory to scan
        
    Returns:
        List of document metadata dictionaries
    """
    documents = find_documents(root_dir)
    return [get_document_info(doc) for doc in documents]


if __name__ == "__main__":
    # Test the scanner
    print("Scanning for documents...")
    docs = scan_workspace()
    print(f"\nFound {len(docs)} documents:")
    for doc in docs:
        print(f"  - {doc['relative_path']} ({doc['size_bytes']} bytes)")
