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
class TradingMethod:
    def __init__(self, config_selector, pricing_strategy):
        self.config_selector = config_selector
        self.pricing_strategy = pricing_strategy
        self.name = f"{config_selector.name} + {pricing_strategy.name}"
        
    def run_step(self, query_cluster: int, t: int, query, vector_db, demand_func):
        ef = self.config_selector.select_configuration(query_cluster, t)
        config_quality = np.sqrt(ef / 256) 
        price = self.pricing_strategy.select_price(query_cluster, ef, t)
        result = vector_db.search(query, k=20, ef=ef)
        demand = demand_func.get_demand(price, config_quality)
        revenue = price * demand
        reward = revenue - result.cost
        self.config_selector.update(query_cluster, ef, reward)
        self.pricing_strategy.update(query_cluster, ef, price, demand)
        
        return {
            'ef': ef,
            'price': price,
            'demand': demand,
            'reward': reward,
            'revenue': revenue,
            'cost': result.cost,
            'config_quality': config_quality
        }