"""
Optimized ConceptClustering.py
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

from src.core.Logger.DarwinLogger import darwin, LogType


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype(np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    return vectors / np.clip(norms, 1e-12, None)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Optimized cosine similarity for pre-normalized vectors.
    """
    return float(np.dot(a, b))


def cluster_chunks(
    chunk_vectors: np.ndarray,
    use_umap: bool = False,
    umap_n_components: int = 10,
    min_cluster_size: int = None,
    min_samples: int = 1,
) -> Tuple[np.ndarray, Dict[int, List[int]]]:

    n_chunks, dim = chunk_vectors.shape

    if min_cluster_size is None:
        min_cluster_size = max(2, n_chunks // 20)

    vectors = normalize_vectors(chunk_vectors)

    enable_umap = use_umap or n_chunks > 300

    if enable_umap:
        try:
            import umap

            n_components = min(
                umap_n_components if n_chunks < 2000 else 15,
                dim,
                n_chunks - 2,
            )

            n_neighbors = min(30, max(2, n_chunks // 10))

            reducer = umap.UMAP(
                n_components=n_components,
                metric="cosine",
                random_state=42,
                min_dist=0.05,
                n_neighbors=n_neighbors,
                n_jobs=1,
            )

            vectors = reducer.fit_transform(vectors)

            darwin.log(
                f"UMAP reduced to {n_components}d for clustering",
                LogType.INFO
            )

        except ImportError:
            darwin.log(
                "UMAP not available, using raw vectors",
                LogType.WARNING
            )

        except Exception as e:
            darwin.log(
                f"UMAP failed: {e}",
                LogType.WARNING
            )

    labels = None

    try:
        import hdbscan

        base = min_cluster_size

        if n_chunks > 1000:
            base = max(min_cluster_size, n_chunks // 50)
            effective_min_samples = max(1, n_chunks // 200)

        else:
            effective_min_samples = min_samples

        retry_sizes = [
            base,
            max(2, base // 2),
            max(2, base // 4)
        ]

        retry_methods = [
            "eom",
            "eom",
            "leaf"
        ]

        for attempt, (mcs, method) in enumerate(
            zip(retry_sizes, retry_methods)
        ):

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=int(mcs),
                min_samples=int(
                    effective_min_samples if attempt == 0 else 1
                ),
                metric="euclidean",
                cluster_selection_method=method,
                cluster_selection_epsilon=0.0,
            )

            labels = clusterer.fit_predict(vectors)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            noise_count = int(np.sum(labels == -1))

            if n_clusters >= 3 or attempt == len(retry_sizes) - 1:

                darwin.log(
                    f"HDBSCAN found {n_clusters} clusters "
                    f"({noise_count} noise chunks)",
                    LogType.INFO
                )

                break

    except Exception as e:

        darwin.log(
            f"HDBSCAN failed ({type(e).__name__}: {e})",
            LogType.WARNING
        )

    if labels is None:

        try:
            from sklearn.cluster import KMeans

            k = min(15, max(2, n_chunks // 5))

            kmeans = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )

            labels = kmeans.fit_predict(vectors)

            darwin.log(
                f"KMeans fallback with k={k}",
                LogType.INFO
            )

        except Exception as e2:

            darwin.log(
                f"KMeans also failed ({e2})",
                LogType.ERROR
            )

            return np.full(n_chunks, -1, dtype=int), {}

    cluster_map: Dict[int, List[int]] = defaultdict(list)

    for idx, label in enumerate(labels):

        if label != -1:
            cluster_map[int(label)].append(idx)

    return labels, dict(cluster_map)
