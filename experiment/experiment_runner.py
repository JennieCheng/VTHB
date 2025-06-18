from experiment.config import ExperimentConfig
from experiment.demand import DeterministicDemandFunction
from experiment.vectordb import VectorDatabase
from experiment.config_selectors import UCBConfigurationSelector, FixedConfigurationSelector, RandomConfigurationSelector
from experiment.pricing_strategies import (
    PricingUCB,
    FixedPricingStrategy,
    RandomPricingStrategy,
    LinearDemandPricing,
    ConvexDemandPricing
)
from experiment.trading_method import TradingMethod
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

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, data=None, query=None):
        self.config = config
        self.config.dim=data.shape[1]
        self.config.num_elements=data.shape[0]
        self.config.num_queries=query.shape[0]

        self.setup_environment(data, query)
        self.setup_methods()
        
    def setup_environment(self, data, query):
        np.random.seed(42)

        self.data=data
        self.queries=query
        
        self.vector_db = VectorDatabase(self.config.dim, self.config.num_elements)
        self.vector_db.add_items(self.data)
        self.vector_db.compute_ground_truth(self.queries, max(self.config.k_values))
        
        self.demand_functions = {
            'unimodal': DeterministicDemandFunction('unimodal'),
            'bimodal': DeterministicDemandFunction('bimodal'),
            'multimodal': DeterministicDemandFunction('multimodal')
        }
    
    def setup_methods(self):
        ucb_config = UCBConfigurationSelector(self.config)
        fixed_config = FixedConfigurationSelector(fixed_ef=64)
        random_config = RandomConfigurationSelector(self.config)
        
        pricing_ucb = PricingUCB(self.config)
        fixed_pricing = FixedPricingStrategy(fixed_price=5.0)
        random_pricing = RandomPricingStrategy(self.config)
        linear_pricing = LinearDemandPricing(self.config)
        convex_pricing = ConvexDemandPricing(self.config)

        self.methods = {
            'VTHB*': TradingMethod(
                UCBConfigurationSelector(self.config), 
                PricingUCB(self.config)
            ),
            
            'STCF ': TradingMethod(
                FixedConfigurationSelector(fixed_ef=64), 
                PricingUCB(self.config)
            ),
            'RDCF': TradingMethod(
                RandomConfigurationSelector(self.config), 
                PricingUCB(self.config)
            ),
            
            'STP': TradingMethod(
                UCBConfigurationSelector(self.config), 
                fixed_pricing
            ),
            'RDP': TradingMethod(
                UCBConfigurationSelector(self.config), 
                random_pricing
            ),
            'LinP': TradingMethod(
                UCBConfigurationSelector(self.config), 
                linear_pricing
            ),
            'ConP': TradingMethod(
                UCBConfigurationSelector(self.config), 
                convex_pricing
            ),
        }
    
    def compute_optimal_reward(self, query: np.ndarray, demand_func: DeterministicDemandFunction) -> float:
        best_reward = -np.inf
        
        for ef in [16, 32, 64, 128, 256]:
            config_quality = np.sqrt(ef / 256)
            result = self.vector_db.search(query, k=20, ef=ef)
            
            optimal_price = demand_func.get_optimal_price(config_quality)
            optimal_demand = demand_func.get_expected_demand(optimal_price, config_quality)
            optimal_revenue = optimal_price * optimal_demand
            optimal_reward = optimal_revenue - result.cost
            
            if optimal_reward > best_reward:
                best_reward = optimal_reward
        
        return best_reward
    
    def run_comparison_experiment(self, T: int = 500, demand_type: str = 'bimodal'):
        results = {}
        demand_func = self.demand_functions[demand_type]
        
        base_seed = 42
        
        for idx, (method_name, method) in enumerate(self.methods.items()):
            print(f"Running {method_name}...")
            np.random.seed(base_seed + idx)
            
            regrets = []
            revenues = []
            cumulative_regret = 0
            cumulative_revenue = 0
            
            for t in range(T):
                query_idx = t % len(self.queries)
                query = self.queries[query_idx]
                query_cluster = query_idx % 5  # 5个聚类
                
                # 执行一步交易
                step_result = method.run_step(query_cluster, t + 1, query, 
                                            self.vector_db, demand_func)
                
                # 计算最优奖励
                optimal_reward = self.compute_optimal_reward(query, demand_func)
                
                # 计算regret
                regret = max(0, optimal_reward - step_result['reward'])
                cumulative_regret += regret
                cumulative_revenue += step_result['revenue']
                
                regrets.append(cumulative_regret)
                revenues.append(cumulative_revenue / (t + 1))
            
            results[method_name] = {
                'regrets': regrets,
                'revenues': revenues,
                'final_regret': cumulative_regret,
                'final_avg_revenue': cumulative_revenue / T
            }
        
        return results
    