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

class DeterministicDemandFunction:
    def __init__(self, modal_type: str = 'unimodal'):
        self.modal_type = modal_type
        self.setup_demand_function()
        self.noise_level = 0.02  # 很小的噪声
        
    def setup_demand_function(self):
        if self.modal_type == 'unimodal':
            self.optimal_price = 5.0
            self.demand_func = lambda p: 0.8 * np.exp(-0.5 * ((p - 5.0) / 2.0) ** 2)
            
        elif self.modal_type == 'bimodal':
            self.optimal_prices = [5.0, 7.0]
            self.demand_func = lambda p: (0.5 * np.exp(-0.5 * ((p - 2.0) / 1.5) ** 2) + 
                                         0.5 * np.exp(-0.5 * ((p - 7.0) / 1.5) ** 2))
            
        elif self.modal_type == 'multimodal':
            self.optimal_prices = [2.0, 5.0, 8.0]
            self.demand_func = lambda p: (0.3 * np.exp(-0.5 * ((p - 2.0) / 1.0) ** 2) + 
                                         0.5 * np.exp(-0.5 * ((p - 5.0) / 1.5) ** 2) + 
                                         0.2 * np.exp(-0.5 * ((p - 8.0) / 1.0) ** 2))
    
    def get_demand(self, price: float, config_quality: float) -> float:
        base_demand = self.demand_func(price)
        
        demand = base_demand * config_quality
        
        noise = np.random.normal(0, self.noise_level)
        
        return np.clip(demand + noise, 0, 1)
    
    def get_expected_demand(self, price: float, config_quality: float) -> float:
        base_demand = self.demand_func(price)
        return base_demand * config_quality
    
    def get_optimal_price(self, config_quality: float) -> float:
        def revenue(p):
            demand = self.get_expected_demand(p, config_quality)
            return -p * demand  
        
        result = minimize_scalar(revenue, bounds=(0.1, 10.0), method='bounded')
        return result.x
