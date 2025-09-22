from __future__ import annotations

import numpy as np
from typing import Optional

try:
    from sklearn.cluster import KMeans
except Exception:  
    KMeans = None


class BaseClusterer:
    def __init__(self, n_clusters: int):
        self.n_clusters = int(n_clusters)

    def fit(self, queries: np.ndarray, data: Optional[np.ndarray] = None):
        return self

    def assign(self, query: np.ndarray) -> int:
        raise NotImplementedError


class RoundRobinClusterer(BaseClusterer):
    def __init__(self, n_clusters: int):
        super().__init__(n_clusters)
        self._counter = 0

    def assign(self, query: np.ndarray) -> int:
        cid = self._counter % self.n_clusters
        self._counter += 1
        return int(cid)


class NormBucketClusterer(BaseClusterer):
    def __init__(self, n_clusters: int):
        super().__init__(n_clusters)
        self._edges: Optional[np.ndarray] = None

    def fit(self, queries: np.ndarray, data: Optional[np.ndarray] = None):
        norms = np.linalg.norm(queries, axis=1)
        # Equal-width bins across observed range
        eps = 1e-12
        min_v, max_v = norms.min(), norms.max() + eps
        self._edges = np.linspace(min_v, max_v, self.n_clusters + 1)
        return self

    def assign(self, query: np.ndarray) -> int:
        assert self._edges is not None, "Clusterer not fit() yet"
        val = np.linalg.norm(query)
        # bin index in [0, n_clusters-1]
        idx = np.searchsorted(self._edges, val, side="right") - 1
        idx = int(np.clip(idx, 0, self.n_clusters - 1))
        return idx


class KMeansClusterer(BaseClusterer):

    def __init__(self, n_clusters: int, random_state: int = 42, max_iter: int = 300):
        super().__init__(n_clusters)
        self.random_state = random_state
        self.max_iter = max_iter
        self._kmeans: Optional[KMeans] = None

    def fit(self, queries: np.ndarray, data: Optional[np.ndarray] = None):
        if KMeans is None:
            raise ImportError("scikit-learn is required for KMeansClusterer")
        self._kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10, max_iter=self.max_iter)
        self._kmeans.fit(queries)
        return self

    def assign(self, query: np.ndarray) -> int:
        assert self._kmeans is not None, "Clusterer not fit() yet"
        cid = int(self._kmeans.predict(query.reshape(1, -1))[0])
        return cid


class IVFLikeClusterer(BaseClusterer):
    def __init__(self, n_clusters: int, random_state: int = 42, max_iter: int = 300):
        super().__init__(n_clusters)
        self.random_state = random_state
        self.max_iter = max_iter
        self._centroids: Optional[np.ndarray] = None

    def fit(self, queries: np.ndarray, data: Optional[np.ndarray] = None):
        if KMeans is None:
            raise ImportError("scikit-learn is required for IVFLikeClusterer")
        if data is None:
            # Fall back to queries if base data not provided
            data = queries
        km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10, max_iter=self.max_iter)
        km.fit(data)
        self._centroids = km.cluster_centers_
        return self

    def assign(self, query: np.ndarray) -> int:
        assert self._centroids is not None, "Clusterer not fit() yet"
        # Nearest centroid by L2
        dists = np.linalg.norm(self._centroids - query, axis=1)
        return int(np.argmin(dists))


class PaperClusterer(BaseClusterer):
    def __init__(self, total_clusters: int = 8, random_state: int = 42, ivf_train: str = 'base'):
        assert total_clusters in (2, 4, 8, 16, 32, 64, 128), "total_clusters must be one of 2,4,8,16,32,64,128"
        super().__init__(n_clusters=total_clusters)
        self.random_state = random_state
        assert ivf_train in ('base', 'query', 'both'), "ivf_train must be 'base'|'query'|'both'"
        self.ivf_train = ivf_train
        # Determine IVF k and whether to use norm/angle splits
        if total_clusters == 2:
            self.ivf_k = 2; self.use_norm = False; self.use_angle = False
        elif total_clusters == 4:
            self.ivf_k = 2; self.use_norm = True;  self.use_angle = False
        elif total_clusters == 8:
            self.ivf_k = 2; self.use_norm = True;  self.use_angle = True
        elif total_clusters in (16, 32, 64, 128):
            self.ivf_k = total_clusters // 4
            self.use_norm = True
            self.use_angle = True
        
        self._ivf_centroids: Optional[np.ndarray] = None
        self._norm_threshold: Optional[float] = None
        self._angle_threshold: Optional[float] = None
        self._data_mean: Optional[np.ndarray] = None

    def fit(self, queries: np.ndarray, data: Optional[np.ndarray] = None):
        self._data_mean = (data.mean(axis=0) if data is not None else queries.mean(axis=0)).astype(float)
        # IVF centroids
        if KMeans is None:
            raise ImportError("scikit-learn is required for PaperClusterer")
        # Choose training source for IVF centroids
        if self.ivf_train == 'base':
            base = data if data is not None else queries
        elif self.ivf_train == 'query':
            base = queries
        else:  # both
            base = queries if data is None else np.vstack([data, queries])
        km = KMeans(n_clusters=self.ivf_k, random_state=self.random_state, n_init=10)
        km.fit(base)
        self._ivf_centroids = km.cluster_centers_
        # Norm split
        if self.use_norm:
            norms = np.linalg.norm(queries, axis=1)
            self._norm_threshold = float(np.median(norms))
        # Angle split
        if self.use_angle:
            qn = np.linalg.norm(queries, axis=1) + 1e-12
            dm = self._data_mean / (np.linalg.norm(self._data_mean) + 1e-12)
            cos_sim = (queries @ dm) / qn
            self._angle_threshold = float(np.median(cos_sim))
        return self

    def assign(self, query: np.ndarray) -> int:
        # IVF id
        assert self._ivf_centroids is not None
        dists = np.linalg.norm(self._ivf_centroids - query, axis=1)
        ivf_id = int(np.argmin(dists))  # in [0, ivf_k-1]

        # Compose with norm/angle bits as Cartesian product
        mult = self.ivf_k
        idx = ivf_id
        if self.use_norm:
            assert self._norm_threshold is not None
            norm_bit = int(np.linalg.norm(query) >= self._norm_threshold)  # 0/1
            idx = ivf_id + mult * norm_bit
            if self.use_angle:
                assert self._angle_threshold is not None and self._data_mean is not None
                dm = self._data_mean / (np.linalg.norm(self._data_mean) + 1e-12)
                ang_bit = int(((query @ dm) / (np.linalg.norm(query) + 1e-12)) >= self._angle_threshold)
                idx = ivf_id + mult * (norm_bit + 2 * ang_bit)
        return idx
