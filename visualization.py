"""
Visualization Module
Provides 2D PCA and t-SNE visualization of vector space.
Shows chunk clusters labeled by source filenames.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import json

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_dimensions(
    vectors: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    random_state: int = 42,
    perplexity: float = 30.0
) -> np.ndarray:
    """
    Reduce high-dimensional vectors to 2D using PCA or t-SNE.

    Args:
        vectors: Array of shape [n_samples, n_features]
        method: 'pca' or 'tsne'
        n_components: Number of dimensions to reduce to (default 2)
        random_state: Random seed for reproducibility
        perplexity: Perplexity parameter for t-SNE

    Returns:
        Reduced vectors of shape [n_samples, n_components]
    """
    if vectors.shape[0] < 2:
        raise ValueError("Need at least 2 vectors for dimensionality reduction")

    # Handle case where we have fewer samples than n_components
    n_samples = vectors.shape[0]
    if n_samples < n_components:
        n_components = n_samples

    if method.lower() == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(vectors)

    elif method.lower() == "tsne":
        # Adjust perplexity for small datasets
        adjusted_perplexity = min(perplexity, n_samples - 1)
        if adjusted_perplexity < 1:
            adjusted_perplexity = 1

        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=adjusted_perplexity,
            n_iter=1000,
            learning_rate='auto',
            init='pca'
        )
        return reducer.fit_transform(vectors)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'")


def get_unique_colors(n_colors: int) -> List[str]:
    """
    Generate a list of distinct colors for plotting.

    Args:
        n_colors: Number of colors needed

    Returns:
        List of hex color codes
    """
    # Predefined color palette (distinct colors)
    palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
        '#5254a3', '#8ca252', '#bd9e39', '#ad494a', '#a55194'
    ]

    # If we need more colors, generate additional ones
    if n_colors > len(palette):
        import random
        random.seed(42)
        while len(palette) < n_colors:
            # Generate random colors
            color = '#{:06x}'.format(random.randint(0, 0xFFFFFF))
            if color not in palette:
                palette.append(color)

    return palette[:n_colors]


def create_2d_scatter_plot(
    reduced_vectors: np.ndarray,
    labels: List[str],
    filenames: List[str],
    texts: List[str],
    title: str = "Document Vector Space Visualization",
    method: str = "PCA"
) -> Dict[str, Any]:
    """
    Create a 2D scatter plot data structure for visualization.

    Args:
        reduced_vectors: 2D vectors [n_samples, 2]
        labels: Source labels for each point
        filenames: Source filenames for each point
        texts: Text content for hover info
        title: Plot title
        method: Dimensionality reduction method used

    Returns:
        Dictionary with plot data for Plotly
    """
    # Get unique filenames for coloring
    unique_filenames = sorted(list(set(filenames)))
    color_map = {fn: i for i, fn in enumerate(unique_filenames)}
    colors = get_unique_colors(len(unique_filenames))

    # Create traces for each unique filename
    traces = []
    for i, filename in enumerate(unique_filenames):
        # Get indices for this filename
        indices = [j for j, fn in enumerate(filenames) if fn == filename]

        trace_x = [reduced_vectors[j, 0] for j in indices]
        trace_y = [reduced_vectors[j, 1] for j in indices]
        trace_texts = [texts[j][:200] + "..." if len(texts[j]) > 200 else texts[j] for j in indices]
        trace_labels = [labels[j] for j in indices]

        traces.append({
            'x': trace_x,
            'y': trace_y,
            'mode': 'markers+text',
            'type': 'scatter',
            'name': filename,
            'text': trace_labels,
            'textposition': 'top center',
            'marker': {
                'size': 12,
                'color': colors[i],
                'opacity': 0.7,
                'line': {'width': 1, 'color': 'white'}
            },
            'hovertemplate': (
                '<b>%{text}</b><br>' +
                f'<b>File:</b> {filename}<br>' +
                '<b>Text:</b> {customdata}<br>' +
                '<extra></extra>'
            ),
            'customdata': trace_texts
        })

    # Layout configuration
    layout = {
        'title': {
            'text': f"{title}<br><sub>Using {method} dimensionality reduction</sub>",
            'font': {'size': 16}
        },
        'xaxis': {
            'title': f'{method} Component 1',
            'showgrid': True,
            'gridwidth': 1,
            'gridcolor': '#e0e0e0'
        },
        'yaxis': {
            'title': f'{method} Component 2',
            'showgrid': True,
            'gridwidth': 1,
            'gridcolor': '#e0e0e0'
        },
        'hovermode': 'closest',
        'showlegend': True,
        'legend': {
            'title': {'text': 'Source Files'},
            'orientation': 'v',
            'yanchor': 'top',
            'y': 1,
            'xanchor': 'left',
            'x': 1.02
        },
        'plot_bgcolor': '#fafafa',
        'paper_bgcolor': 'white',
        'width': 900,
        'height': 600,
        'margin': {'l': 60, 'r': 150, 't': 80, 'b': 60}
    }

    return {
        'data': traces,
        'layout': layout
    }


def visualize_vector_store(
    vector_store,
    method: str = "pca",
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create visualization from a VectorStore instance.

    Args:
        vector_store: VectorStore instance
        method: 'pca' or 'tsne'
        sample_size: Optional limit on number of vectors to visualize

    Returns:
        Plotly figure data
    """
    if vector_store.is_empty():
        raise ValueError("Vector store is empty")

    # Get all embeddings
    embeddings = vector_store.get_all_embeddings()
    metadata = vector_store.get_all_metadata()

    # Sample if needed
    if sample_size and len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[indices]
        metadata = [metadata[i] for i in indices]

    # Extract labels and filenames
    labels = []
    filenames = []
    texts = []

    for meta in metadata:
        source_file = meta.get('source_file', 'Unknown')
        chunk_index = meta.get('chunk_index', 0)
        text = meta.get('text', '')

        filenames.append(source_file)
        labels.append(f"Chunk {chunk_index + 1}")
        texts.append(text)

    # Reduce dimensions
    reduced = reduce_dimensions(embeddings, method=method)

    # Create plot
    return create_2d_scatter_plot(
        reduced_vectors=reduced,
        labels=labels,
        filenames=filenames,
        texts=texts,
        title="Knowledge Base Vector Space",
        method=method.upper()
    )


def plot_clusters(
    vectors: np.ndarray,
    labels: List[str],
    filenames: List[str],
    texts: List[str],
    method: str = "pca"
) -> Dict[str, Any]:
    """
    Create a 2D plot of vector clusters.

    Args:
        vectors: High-dimensional vectors [n_samples, n_features]
        labels: Labels for each point
        filenames: Source filenames for coloring
        texts: Text content for hover
        method: 'pca' or 'tsne'

    Returns:
        Plotly figure data
    """
    # Reduce dimensions
    reduced = reduce_dimensions(vectors, method=method)

    # Create plot
    return create_2d_scatter_plot(
        reduced_vectors=reduced,
        labels=labels,
        filenames=filenames,
        texts=texts,
        method=method.upper()
    )


def get_visualization_stats(
    vector_store
) -> Dict[str, Any]:
    """
    Get statistics for visualization.

    Args:
        vector_store: VectorStore instance

    Returns:
        Dictionary with statistics
    """
    if vector_store.is_empty():
        return {
            'total_vectors': 0,
            'unique_files': 0,
            'files': []
        }

    metadata = vector_store.get_all_metadata()
    filenames = [meta.get('source_file', 'Unknown') for meta in metadata]
    unique_files = sorted(list(set(filenames)))

    return {
        'total_vectors': len(metadata),
        'unique_files': len(unique_files),
        'files': unique_files
    }


if __name__ == "__main__":
    # Test visualization module
    print("Testing Visualization Module...\n")

    # Create sample data
    n_samples = 50
    n_features = 384

    # Generate random embeddings
    np.random.seed(42)
    vectors = np.random.randn(n_samples, n_features).astype(np.float32)

    # Create sample metadata
    filenames = ['doc_a.pdf'] * 20 + ['doc_b.md'] * 20 + ['doc_c.txt'] * 10
    labels = [f"Chunk {i+1}" for i in range(n_samples)]
    texts = [f"Sample text content for chunk {i+1}" for i in range(n_samples)]

    print(f"Created {n_samples} sample vectors")
    print(f"Vector shape: {vectors.shape}")
    print(f"Unique files: {set(filenames)}")

    # Test PCA
    print("\nTesting PCA reduction...")
    try:
        reduced_pca = reduce_dimensions(vectors, method="pca")
        print(f"PCA reduced shape: {reduced_pca.shape}")
        print("PCA test passed!")
    except Exception as e:
        print(f"PCA test failed: {e}")

    # Test t-SNE
    print("\nTesting t-SNE reduction...")
    try:
        reduced_tsne = reduce_dimensions(vectors, method="tsne")
        print(f"t-SNE reduced shape: {reduced_tsne.shape}")
        print("t-SNE test passed!")
    except Exception as e:
        print(f"t-SNE test failed: {e}")

    # Test plot creation
    print("\nTesting plot creation...")
    try:
        plot_data = create_2d_scatter_plot(
            reduced_vectors=reduced_pca,
            labels=labels,
            filenames=filenames,
            texts=texts,
            method="PCA"
        )
        print(f"Plot data keys: {plot_data.keys()}")
        print(f"Number of traces: {len(plot_data['data'])}")
        print(f"Plot title: {plot_data['layout']['title']['text']}")
        print("Plot creation test passed!")
    except Exception as e:
        print(f"Plot creation test failed: {e}")

    print("\nVisualization module test complete!")
