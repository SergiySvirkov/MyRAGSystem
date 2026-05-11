"""
Text Extractor Module
Extracts text content from PDF and Markdown files.
"""

import re
from pathlib import Path
from typing import Optional


def extract_from_pdf(file_path: Path) -> str:
    """
    Extract text from a PDF file using PyPDF2.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text content
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError("PyPDF2 is required. Install with: pip install PyPDF2")
    
    text = ""
    try:
        reader = PdfReader(str(file_path))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""
    
    return clean_text(text)


def extract_from_markdown(file_path: Path) -> str:
    """
    Extract text from a Markdown file.
    
    Args:
        file_path: Path to the Markdown file
        
    Returns:
        Text content (with minimal markdown formatting preserved)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading Markdown {file_path}: {e}")
        return ""
    
    return clean_text(content)


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing extra whitespace and normalizing.
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Remove excessive whitespace
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


def extract_text(file_path: Path) -> Optional[str]:
    """
    Extract text from a document based on its file extension.
    
    Args:
        file_path: Path to the document
        
    Returns:
        Extracted text or None if extraction failed
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None
    
    extension = file_path.suffix.lower()
    
    if extension == '.pdf':
        return extract_from_pdf(file_path)
    elif extension in {'.md', '.markdown'}:
        return extract_from_markdown(file_path)
    else:
        print(f"Unsupported file type: {extension}")
        return None


def get_document_stats(text: str) -> dict:
    """
    Get statistics about extracted text.
    
    Args:
        text: Extracted text content
        
    Returns:
        Dictionary with text statistics
    """
    if not text:
        return {
            'char_count': 0,
            'word_count': 0,
            'line_count': 0
        }
    
    return {
        'char_count': len(text),
        'word_count': len(text.split()),
        'line_count': len(text.split('\n'))
    }


if __name__ == "__main__":
    # Test the extractor
    import sys
    from document_scanner import find_documents
    
    docs = find_documents()
    print(f"Testing text extraction on {len(docs)} documents...\n")
    
    for doc in docs[:3]:  # Test first 3 documents
        print(f"Extracting: {doc.name}")
        text = extract_text(doc)
        if text:
            stats = get_document_stats(text)
            print(f"  Characters: {stats['char_count']}")
            print(f"  Words: {stats['word_count']}")
            print(f"  Lines: {stats['line_count']}")
            print(f"  Preview: {text[:200]}...")
        print()
