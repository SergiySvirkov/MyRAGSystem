#!/usr/bin/env python3
"""
Main CLI Script for Document RAG System
Handles document indexing, re-indexing, and launches the web app.
Supports multilingual embeddings and LLM integration.
"""

import argparse
import os
import sys
import time
import shutil
from pathlib import Path

# Ensure all modules can be imported
sys.path.insert(0, str(Path(__file__).parent))

from document_scanner import scan_workspace
from text_extractor import extract_text
from chunker import chunk_multiple_documents, get_chunk_statistics
from embeddings import get_embedding_model, DEFAULT_MODEL
from vector_store import create_vector_store_from_chunks, reindex_documents
from search import DocumentSearch, create_search_engine


def index_documents(
    root_dir: str = ".",
    index_path: str = "./vector_index",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    model_name: str = DEFAULT_MODEL,
    clear_existing: bool = False
):
    """
    Index all documents in the workspace.

    Args:
        root_dir: Root directory to scan
        index_path: Path to save the index
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        model_name: Embedding model name
        clear_existing: Whether to clear existing index first
    """
    print("=" * 60)
    print("DOCUMENT INDEXING")
    print(f"Model: {model_name}")
    print("=" * 60)

    # Clear existing index if requested
    if clear_existing and os.path.exists(index_path):
        print(f"\n🗑️  Clearing existing index at {index_path}...")
        shutil.rmtree(index_path)
        print("   Existing index cleared")

    # Step 1: Scan for documents
    print("\n📁 Step 1: Scanning for documents...")
    start_time = time.time()

    doc_infos = scan_workspace(root_dir)

    if not doc_infos:
        print("❌ No PDF or Markdown files found!")
        return False

    print(f"✅ Found {len(doc_infos)} document(s):")
    for doc in doc_infos:
        print(f"   - {doc['relative_path']} ({doc['size_bytes']:,} bytes)")

    # Step 2: Extract text
    print("\n📄 Step 2: Extracting text...")
    documents = []

    for doc_info in doc_infos:
        file_path = Path(doc_info['path'])
        print(f"   Extracting: {file_path.name}...", end=" ")

        text = extract_text(file_path)
        if text:
            documents.append((text, str(file_path)))
            print(f"✓ ({len(text):,} chars)")
        else:
            print("✗ (failed)")

    if not documents:
        print("❌ No text could be extracted from documents!")
        return False

    print(f"✅ Successfully extracted text from {len(documents)} document(s)")

    # Step 3: Chunk documents
    print("\n✂️  Step 3: Chunking documents...")

    chunks = chunk_multiple_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    if not chunks:
        print("❌ No chunks created!")
        return False

    stats = get_chunk_statistics(chunks)
    print(f"✅ Created {stats['total_chunks']} chunks:")
    print(f"   - Average size: {stats['avg_chunk_size']:.0f} chars")
    print(f"   - Min size: {stats['min_chunk_size']} chars")
    print(f"   - Max size: {stats['max_chunk_size']} chars")
    print(f"   - Unique sources: {stats['unique_sources']}")

    # Step 4: Create embeddings
    print("\n🧠 Step 4: Creating embeddings...")
    print(f"   Model: {model_name}")

    embedding_model = get_embedding_model(model_name)
    embedding_model.load()

    print(f"   Embedding dimension: {embedding_model.embedding_dim}")
    print(f"   Encoding {len(chunks)} chunks...")

    embeddings = embedding_model.encode_chunks(chunks)
    print(f"✅ Created {len(embeddings)} embeddings")

    # Step 5: Build vector store
    print("\n💾 Step 5: Building vector store...")

    vector_store = create_vector_store_from_chunks(
        chunks,
        embedding_model,
        index_path=index_path
    )

    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print("INDEXING COMPLETE")
    print(f"{'=' * 60}")
    print(f"📊 Statistics:")
    print(f"   - Documents processed: {len(documents)}")
    print(f"   - Chunks created: {len(chunks)}")
    print(f"   - Vectors indexed: {vector_store.index.ntotal}")
    print(f"   - Index saved to: {index_path}")
    print(f"   - Time elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}\n")

    return True


def reindex_documents_cmd(
    root_dir: str = ".",
    index_path: str = "./vector_index",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    model_name: str = DEFAULT_MODEL,
    backup: bool = True
):
    """
    Re-index all documents with the current embedding model.

    Args:
        root_dir: Root directory to scan
        index_path: Path to save the index
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        model_name: Embedding model name
        backup: Whether to backup the old index
    """
    print("=" * 60)
    print("RE-INDEXING DOCUMENTS")
    print(f"Model: {model_name}")
    print("=" * 60)

    # Backup existing index
    if backup and os.path.exists(index_path):
        backup_path = str(index_path) + "_backup_" + time.strftime("%Y%m%d_%H%M%S")
        print(f"\n💾 Backing up existing index to {backup_path}...")
        shutil.copytree(index_path, backup_path)
        print("   Backup complete")

    # Run indexing with clear_existing=True
    return index_documents(
        root_dir=root_dir,
        index_path=index_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_name=model_name,
        clear_existing=True
    )


def test_search(
    index_path: str = "./vector_index",
    llm_provider: str = None,
    llm_model: str = None
):
    """Test the search functionality."""
    print("\n" + "=" * 60)
    print("TESTING SEARCH")
    print("=" * 60)

    if not os.path.exists(index_path):
        print(f"❌ Index not found at {index_path}")
        print("   Run indexing first: python main.py --index")
        return False

    try:
        search = create_search_engine(index_path, llm_provider, llm_model)

        print("\n✅ Search engine loaded successfully")
        print(f"   Index path: {index_path}")
        print(f"   Total vectors: {search.vector_store.index.ntotal}")

        if search.llm_integration:
            print(f"   LLM: {search.llm_integration.provider_name}")

        # Test queries
        test_queries = [
            "What is this document about?",
            "machine learning",
            "Python programming"
        ]

        print("\n🧪 Running test queries...")
        for query in test_queries:
            print(f"\n   Query: '{query}'")
            results = search.search(query, top_k=2)
            if results:
                print(f"   Found {len(results)} result(s):")
                for r in results:
                    preview = r.text[:100].replace('\n', ' ')
                    print(f"      - {r.source_file} (score: {r.score:.2f})")
                    print(f"        {preview}...")
            else:
                print("   No results found")

        # Test LLM answer if available
        if search.llm_integration and search.llm_integration.is_available():
            print("\n🤖 Testing LLM answer generation...")
            query = "What is machine learning?"
            print(f"   Query: '{query}'")
            response = search.search_with_answer(query, top_k=3, generate_answer=True)
            if response.answer:
                print(f"   Answer: {response.answer[:200]}...")
                print(f"   Tokens used: {response.tokens_used}")
            else:
                print("   No answer generated")

        print("\n✅ Search test completed")
        return True

    except Exception as e:
        print(f"❌ Error testing search: {e}")
        import traceback
        traceback.print_exc()
        return False


def launch_app(index_path: str = "./vector_index", port: int = 8080):
    """Launch the Streamlit web app."""
    print("\n" + "=" * 60)
    print("LAUNCHING WEB APP")
    print("=" * 60)

    if not os.path.exists(index_path):
        print(f"❌ Index not found at {index_path}")
        print("   Run indexing first: python main.py --index")
        return False

    print(f"\n🚀 Starting Streamlit app on port {port}...")
    print(f"   Index path: {index_path}")
    print(f"   URL: http://localhost:{port}")
    print(f"\n   Press Ctrl+C to stop the server\n")

    # Set environment variable for index path
    os.environ['INDEX_PATH'] = index_path

    # Launch Streamlit
    import subprocess

    cmd = [
        "streamlit", "run",
        "app.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.serverAddress", "localhost",
        "--server.enableCORS", "false"
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        return False

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Document RAG System - Index and search your documents with multilingual support and LLM integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
# Index documents in current directory
python main.py --index

# Re-index documents with new multilingual model
python main.py --re-index

# Index documents in specific directory
python main.py --index --root-dir /path/to/docs

# Test search functionality
python main.py --test

# Test search with LLM
python main.py --test --llm-provider ollama

# Launch web app
python main.py --app

# Index and launch app
python main.py --index --app
"""
    )

    parser.add_argument(
        '--index',
        action='store_true',
        help='Index documents in the workspace'
    )

    parser.add_argument(
        '--re-index',
        action='store_true',
        help='Re-index all documents with the current embedding model (replaces existing index)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Test search functionality'
    )

    parser.add_argument(
        '--app',
        action='store_true',
        help='Launch the Streamlit web app'
    )

    parser.add_argument(
        '--root-dir',
        default='.',
        help='Root directory to scan for documents (default: current directory)'
    )

    parser.add_argument(
        '--index-path',
        default='./vector_index',
        help='Path to save/load the FAISS index (default: ./vector_index)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=500,
        help='Size of text chunks in characters (default: 500)'
    )

    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=50,
        help='Overlap between chunks in characters (default: 50)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port for the web app (default: 8080)'
    )

    parser.add_argument(
        '--model',
        default=DEFAULT_MODEL,
        help=f'Sentence-transformers model name (default: {DEFAULT_MODEL})'
    )

    parser.add_argument(
        '--llm-provider',
        choices=['ollama', 'openai'],
        help='LLM provider for answer generation (ollama or openai)'
    )

    parser.add_argument(
        '--llm-model',
        help='LLM model name (provider-specific)'
    )

    parser.add_argument(
        '--openai-api-key',
        help='OpenAI API key (or set OPENAI_API_KEY environment variable)'
    )

    args = parser.parse_args()

    # If no arguments provided, show help
    if not any([args.index, args.re_index, args.test, args.app]):
        parser.print_help()
        print("\n💡 Tip: Use --index to start indexing your documents")
        return

    # Re-index documents
    if args.re_index:
        success = reindex_documents_cmd(
            root_dir=args.root_dir,
            index_path=args.index_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            model_name=args.model,
            backup=True
        )
        if not success:
            sys.exit(1)

    # Index documents
    if args.index:
        success = index_documents(
            root_dir=args.root_dir,
            index_path=args.index_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            model_name=args.model,
            clear_existing=False
        )
        if not success:
            sys.exit(1)

    # Test search
    if args.test:
        # Set OpenAI API key if provided
        if args.openai_api_key:
            os.environ['OPENAI_API_KEY'] = args.openai_api_key

        success = test_search(
            index_path=args.index_path,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model
        )
        if not success:
            sys.exit(1)

    # Launch app
    if args.app:
        success = launch_app(index_path=args.index_path, port=args.port)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
