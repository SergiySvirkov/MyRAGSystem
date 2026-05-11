"""
Streamlit Web Interface for RAG Document Search
Multi-tab layout: Search, Visualization, Settings
Features: Multilingual embeddings, LLM integration, 2D vector visualization
"""

import os
import sys
from pathlib import Path

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import json

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from search import DocumentSearch, SearchResult
from vector_store import VectorStore
from llm_integration import create_llm_integration, LLMIntegration
from visualization import visualize_vector_store, get_visualization_stats

# Page configuration
st.set_page_config(
    page_title="RAG Document Search",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'search_engine' not in st.session_state:
    st.session_state['search_engine'] = None
if 'llm_integration' not in st.session_state:
    st.session_state['llm_integration'] = None
if 'search_history' not in st.session_state:
    st.session_state['search_history'] = []


def load_search_engine():
    """Load or get cached search engine."""
    if st.session_state['search_engine'] is None:
        index_path = st.session_state.get('index_path', './vector_index')

        if not os.path.exists(index_path):
            return None

        try:
            search = DocumentSearch()
            search.load_index(index_path)
            st.session_state['search_engine'] = search
            return search
        except Exception as e:
            st.error(f"Error loading search engine: {e}")
            return None

    return st.session_state['search_engine']


def init_llm_integration():
    """Initialize LLM integration from settings."""
    if st.session_state['llm_integration'] is None:
        provider = st.session_state.get('llm_provider', 'ollama')
        model = st.session_state.get('llm_model', '')
        api_key = st.session_state.get('openai_api_key', '')

        try:
            kwargs = {'provider': provider}
            if model:
                kwargs['model'] = model
            if provider == 'openai' and api_key:
                kwargs['api_key'] = api_key

            llm = create_llm_integration(**kwargs)
            st.session_state['llm_integration'] = llm
            return llm
        except Exception as e:
            st.error(f"Error initializing LLM: {e}")
            return None

    return st.session_state['llm_integration']


def render_search_tab():
    """Render the Search tab."""
    st.header("🔍 Search Documents")
    st.markdown("Ask questions about your documents and get AI-powered answers with source citations.")

    # Load search engine
    search_engine = load_search_engine()

    if search_engine is None:
        st.warning("⚠️ No index found. Please index your documents first.")
        st.info("Run: `python main.py --index` from the terminal")
        return

    # Search settings
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input(
            "Enter your question",
            placeholder="e.g., What is machine learning?",
            key="search_query"
        )
    with col2:
        top_k = st.number_input("Results", min_value=1, max_value=20, value=5)
    with col3:
        use_llm = st.checkbox("Use LLM", value=True, help="Generate AI answer from retrieved context")

    # LLM settings
    if use_llm:
        llm = init_llm_integration()
        if llm and llm.is_available():
            st.success(f"✅ LLM ready ({llm.provider_name})")
        else:
            st.warning("⚠️ LLM not available. Check Settings tab.")
            use_llm = False

    # Search button
    if st.button("🔍 Search", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a search query.")
            return

        with st.spinner("Searching..."):
            try:
                # Perform search
                results = search_engine.search(query, top_k=top_k)

                if not results:
                    st.warning("No results found. Try a different query.")
                    return

                st.success(f"Found {len(results)} relevant chunks")

                # Generate LLM answer if enabled
                if use_llm and llm:
                    with st.spinner("Generating AI answer..."):
                        context_chunks = [r.to_dict() for r in results]
                        llm_response = llm.generate_answer(query, context_chunks)

                        if llm_response.answer:
                            st.subheader("🤖 AI Answer")
                            st.info(llm_response.answer)
                            if llm_response.tokens_used:
                                st.caption(f"Tokens used: {llm_response.tokens_used}")
                        elif llm_response.error:
                            st.error(f"LLM Error: {llm_response.error}")

                # Display source chunks
                st.subheader("📄 Source Chunks")

                for i, result in enumerate(results, 1):
                    with st.expander(f"{i}. {result.source_file} (Relevance: {result.score:.1%})", expanded=(i==1)):
                        st.markdown(f"**Chunk {result.chunk_index + 1} of {result.total_chunks}**")
                        st.markdown(
                            f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>"
                            f"<pre style='white-space: pre-wrap; margin: 0; font-family: inherit; font-size: 14px;'>"
                            f"{result.text}</pre></div>",
                            unsafe_allow_html=True
                        )

                # Save to history
                st.session_state['search_history'].append({
                    'query': query,
                    'num_results': len(results)
                })

            except Exception as e:
                st.error(f"Search error: {e}")
                st.exception(e)


def render_visualization_tab():
    """Render the Visualization tab."""
    st.header("📊 Knowledge Base Visualization")
    st.markdown("Explore the vector space of your documents using 2D projection.")

    # Load search engine
    search_engine = load_search_engine()

    if search_engine is None or search_engine.vector_store.is_empty():
        st.warning("⚠️ No index found. Please index your documents first.")
        return

    # Visualization settings
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        method = st.selectbox("Method", ["PCA", "t-SNE"], index=0)
    with col2:
        sample_size = st.selectbox(
            "Sample Size",
            ["All", "100", "250", "500", "1000"],
            index=0
        )
    with col3:
        if st.button("🔄 Refresh Visualization", type="primary"):
            st.rerun()

    # Show stats
    stats = get_visualization_stats(search_engine.vector_store)
    st.markdown(f"**{stats['total_vectors']}** vectors from **{stats['unique_files']}** documents")

    # Create visualization
    with st.spinner("Generating visualization..."):
        try:
            n_sample = None if sample_size == "All" else int(sample_size)

            plot_data = visualize_vector_store(
                search_engine.vector_store,
                method=method.lower(),
                sample_size=n_sample
            )

            # Create Plotly figure
            fig = go.Figure(data=plot_data['data'], layout=plot_data['layout'])
            st.plotly_chart(fig, use_container_width=True)

            # Show file legend
            st.subheader("📁 Source Files")
            file_cols = st.columns(min(len(stats['files']), 4))
            for i, filename in enumerate(stats['files']):
                with file_cols[i % len(file_cols)]:
                    st.markdown(f"- `{filename}`")

        except Exception as e:
            st.error(f"Visualization error: {e}")
            st.exception(e)


def render_settings_tab():
    """Render the Settings tab."""
    st.header("⚙️ Settings")
    st.markdown("Configure your RAG system preferences.")

    # Index settings
    st.subheader("📂 Index Settings")
    index_path = st.text_input(
        "Index Path",
        value=st.session_state.get('index_path', './vector_index'),
        help="Path to the FAISS index directory"
    )
    st.session_state['index_path'] = index_path

    # Check index status
    if os.path.exists(index_path):
        try:
            store = VectorStore(index_path=index_path)
            stats = store.get_stats()
            st.success(f"✅ Index loaded: {stats['total_vectors']} vectors, {stats['embedding_dim']} dimensions")
        except Exception as e:
            st.error(f"❌ Error loading index: {e}")
    else:
        st.warning("⚠️ Index not found at this path")

    # LLM Settings
    st.subheader("🤖 LLM Settings")

    provider = st.selectbox(
        "LLM Provider",
        ["ollama", "openai"],
        index=0 if st.session_state.get('llm_provider', 'ollama') == 'ollama' else 1,
        help="Choose between local Ollama or OpenAI API"
    )
    st.session_state['llm_provider'] = provider

    if provider == "ollama":
        st.markdown("**Ollama (Local)**")
        ollama_model = st.text_input(
            "Ollama Model",
            value=st.session_state.get('ollama_model', 'llama3.2'),
            help="Model name as shown in 'ollama list'"
        )
        st.session_state['llm_model'] = ollama_model
        st.session_state['ollama_model'] = ollama_model

        # Test Ollama connection
        if st.button("🔄 Test Ollama Connection"):
            try:
                llm = create_llm_integration("ollama", model=ollama_model)
                if llm.is_available():
                    st.success("✅ Ollama is running and accessible!")
                    models = llm.provider.list_models()
                    if models:
                        st.markdown("**Available models:**")
                        for m in models:
                            st.markdown(f"- `{m}`")
                else:
                    st.error("❌ Ollama not accessible. Is the server running?")
                    st.info("Start Ollama: `ollama serve`")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    else:  # openai
        st.markdown("**OpenAI API**")
        openai_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.get('openai_api_key', ''),
            type="password",
            help="Your OpenAI API key"
        )
        st.session_state['openai_api_key'] = openai_key
        os.environ['OPENAI_API_KEY'] = openai_key

        openai_model = st.selectbox(
            "OpenAI Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"],
            index=0
        )
        st.session_state['llm_model'] = openai_model
        st.session_state['openai_model'] = openai_model

        # Test OpenAI connection
        if st.button("🔄 Test OpenAI Connection"):
            if not openai_key:
                st.error("❌ Please enter your OpenAI API key")
            else:
                try:
                    llm = create_llm_integration("openai", api_key=openai_key)
                    if llm.is_available():
                        st.success("✅ OpenAI API key is valid!")
                    else:
                        st.error("❌ API key validation failed")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # Re-index option
    st.subheader("🔄 Re-index Documents")
    st.markdown("Re-index all documents with the current multilingual embedding model.")

    if st.button("🚀 Re-index Now", type="primary"):
        st.warning("Re-indexing will replace the current index. This may take a few minutes.")
        st.info("Run from terminal: `python main.py --re-index`")


def main():
    """Main Streamlit app with multi-tab layout."""

    # Title
    st.title("📚 RAG Document Search")
    st.markdown(
        "**Multilingual Document Search with AI-Powered Answers**"
    )

    # Create tabs
    tab_search, tab_viz, tab_settings = st.tabs([
        "🔍 Search",
        "📊 Visualization",
        "⚙️ Settings"
    ])

    with tab_search:
        render_search_tab()

    with tab_viz:
        render_visualization_tab()

    with tab_settings:
        render_settings_tab()

    # Footer
    st.markdown("---")
    st.caption(
        "Built with ❤️ using Streamlit, sentence-transformers, FAISS, and LLMs | "
        "Embedding Model: paraphrase-multilingual-MiniLM-L12-v2"
    )


if __name__ == "__main__":
    main()
