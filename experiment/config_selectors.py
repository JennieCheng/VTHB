from experiment.config import ExperimentConfig
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
class UCBConfigurationSelector:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.ef_candidates = [16, 32, 64, 128, 256]
        self.config_history = {}
        self.name = "UCB Configuration"
        
    def select_configuration(self, query_cluster: int, t: int) -> int:
        if query_cluster not in self.config_history:
            self.config_history[query_cluster] = {}
        
        for ef in self.ef_candidates:
            if ef not in self.config_history[query_cluster]:
                return ef
        
        best_ucb = -np.inf
        best_ef = self.ef_candidates[0]
        
        for ef in self.ef_candidates:
            stats = self.config_history[query_cluster][ef]
            if stats['count'] == 0:
                return ef
                
            mean_reward = stats['total_reward'] / stats['count']
            exploration_bonus = np.sqrt(2 * np.log(t + 1) / stats['count'])
            ucb = mean_reward + exploration_bonus
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_ef = ef
        
        return best_ef
    
    def update(self, query_cluster: int, ef: int, reward: float):
        if query_cluster not in self.config_history:
            self.config_history[query_cluster] = {}
        if ef not in self.config_history[query_cluster]:
            self.config_history[query_cluster][ef] = {'total_reward': 0, 'count': 0}
        
        self.config_history[query_cluster][ef]['total_reward'] += reward
        self.config_history[query_cluster][ef]['count'] += 1

class FixedConfigurationSelector:
    def __init__(self, fixed_ef: int = 64):
        self.fixed_ef = fixed_ef
        self.name = "Fixed Configuration"
        
    def select_configuration(self, query_cluster: int, t: int) -> int:
        return self.fixed_ef
    
    def update(self, query_cluster: int, ef: int, reward: float):
        pass

class RandomConfigurationSelector:
    def __init__(self, config: ExperimentConfig):
        self.ef_candidates = [16, 32, 64, 128, 256]
        self.name = "Random Configuration"
        
    def select_configuration(self, query_cluster: int, t: int) -> int:
        return np.random.choice(self.ef_candidates)
    
    def update(self, query_cluster: int, ef: int, reward: float):
        pass