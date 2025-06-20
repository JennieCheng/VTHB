from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class ExperimentConfig:
    dim: int = 16
    num_elements: int = 100000
    num_queries: int = 100
    T: int = 500
    price_range: Tuple[float, float] = (0.1, 10.0)
    k_values: List[int] = field(default_factory=lambda: [20])
    c_values: List[float] = field(default_factory=lambda: [1.2])
    ef_range: Tuple[int, int] = (16, 256)
    taylor_order: int = 2
    num_intervals: int = 10
    C_holder: float = 1.0
    delta: float = 0.1
