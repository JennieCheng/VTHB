from dataclasses import dataclass
import numpy as np
import hnswlib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable
import time
from dataclasses import dataclass, field
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, LinearRegression
from scipy.optimize import minimize_scalar
from matplotlib.ticker import ScalarFormatter
import matplotlib.lines as mlines
@dataclass
class QueryResult:
    recall: float
    latency: float
    cost: float
    retrieved_ids: List[int]

class VectorDatabase:
    def __init__(self, dim: int, max_elements: int):
        self.dim = dim
        self.index = hnswlib.Index(space='l2', dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=200, M=16)
        self.data = None
        self.ground_truth = {}
        
    def add_items(self, data: np.ndarray):
        self.data = data
        self.index.add_items(data, np.arange(len(data)))
        
    def compute_ground_truth(self, queries: np.ndarray, k_max: int = 100):
        for i, query in enumerate(queries):
            distances = np.linalg.norm(self.data - query, axis=1)
            self.ground_truth[i] = np.argsort(distances)[:k_max]
    
    def search(self, query: np.ndarray, k: int, ef: int) -> QueryResult:
        self.index.set_ef(ef)
        
        start_time = time.time()
        labels, distances = self.index.knn_query(query, k=k)
        latency = time.time() - start_time
        cost = 0.0001 * ef + 0.001 * ef * np.log(k + 1) + 0.01 * latency
        
        return QueryResult(
            recall=0.0,
            latency=latency,
            cost=cost,
            retrieved_ids=labels[0].tolist()
        )