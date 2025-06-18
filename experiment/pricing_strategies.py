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
class PricingUCB:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.price_history = {}  # {cluster_id: {ef: {interval_stats, data, interval_history}}}
        self.name = "PricingUCB"
        self.k = config.taylor_order
        self.N = config.num_intervals
        self.C = config.C_holder
        self.Delta = config.delta

    def select_price(self, cluster: int, ef: int, t: int) -> float:
        if cluster not in self.price_history:
            self.price_history[cluster] = {}
        if ef not in self.price_history[cluster]:
            self.price_history[cluster][ef] = {
                'interval_stats': {},
                'data': [],
                'interval_history': {}
            }
        intervals = self._partition_price_range()
        selected_interval, interval_idx = self._select_interval_ucb(cluster, ef, intervals, t)
        price = self._subroutine(cluster, ef, selected_interval, interval_idx, t)

        return price

    def _partition_price_range(self):
        return [
            (
                self.config.price_range[0] + i * (self.config.price_range[1] - self.config.price_range[0]) / self.N,
                self.config.price_range[0] + (i + 1) * (self.config.price_range[1] - self.config.price_range[0]) / self.N
            )
            for i in range(self.N)
        ]

    def _select_interval_ucb(self, cluster: int, ef: int, intervals: List[Tuple[float, float]], t: int) -> Tuple[Tuple[float, float], int]:
        best_ucb = -np.inf
        best_interval = intervals[0]
        best_idx = 0

        for idx, interval in enumerate(intervals):
            stats = self.price_history[cluster][ef]['interval_stats'].get(idx, {'total_revenue': 0, 'count': 0})

            if stats['count'] == 0:
                return interval, idx

            avg_revenue = stats['total_revenue'] / stats['count']
            d = self._compute_feature_dimension()
            confidence_radius = 4 * self.config.price_range[1] * np.sqrt(
                2 * d * np.log(d * t + 1)
            ) * (self.Delta + (self.C + np.sqrt(2)) / np.sqrt(stats['count']))

            ucb = avg_revenue + confidence_radius

            if ucb > best_ucb:
                best_ucb = ucb
                best_interval = interval
                best_idx = idx

        return best_interval, best_idx

    def _compute_feature_dimension(self) -> int:
        return self.k + 1

    def _subroutine(self, cluster: int, ef: int, interval: Tuple[float, float], interval_idx: int, t: int) -> float:
        a, b = interval
        if interval_idx not in self.price_history[cluster][ef]['interval_history']:
            self.price_history[cluster][ef]['interval_history'][interval_idx] = []

        interval_data = self.price_history[cluster][ef]['interval_history'][interval_idx]

        if len(interval_data) < 3:
            return np.random.uniform(a, b)

        X = []
        y = []
        for price, demand in interval_data:
            features = self._compute_taylor_features(price, a)
            X.append(features)
            y.append(demand)

        X = np.array(X)
        y = np.array(y)

        lambda_reg = 1.0
        XtX = X.T @ X + lambda_reg * np.eye(X.shape[1])
        if np.linalg.det(XtX) < 1e-10:
            return (a + b) / 2

        Xty = X.T @ y
        theta_hat = np.linalg.solve(XtX, Xty)

        best_price = a
        best_value = -np.inf

        return best_price

    def _compute_taylor_features(self, price: float, reference_point: float) -> np.ndarray:
        delta_p = (price - reference_point) / (self.config.price_range[1] - self.config.price_range[0])
        features = [delta_p ** i for i in range(self.k + 1)]
        return np.array(features)

    def update(self, cluster: int, ef: int, price: float, demand: float):
        if cluster not in self.price_history:
            self.price_history[cluster] = {}
        if ef not in self.price_history[cluster]:
            self.price_history[cluster][ef] = {
                'interval_stats': {},
                'data': [],
                'interval_history': {}
            }
        self.price_history[cluster][ef]['data'].append((price, demand))

        intervals = self._partition_price_range()
        for idx, (a, b) in enumerate(intervals):
            if a <= price <= b:
                # 更新区间统计
                if idx not in self.price_history[cluster][ef]['interval_stats']:
                    self.price_history[cluster][ef]['interval_stats'][idx] = {'total_revenue': 0, 'count': 0}

                revenue = price * demand
                self.price_history[cluster][ef]['interval_stats'][idx]['total_revenue'] += revenue
                self.price_history[cluster][ef]['interval_stats'][idx]['count'] += 1

                # 保存到区间历史
                if idx not in self.price_history[cluster][ef]['interval_history']:
                    self.price_history[cluster][ef]['interval_history'][idx] = []
                self.price_history[cluster][ef]['interval_history'][idx].append((price, demand))
                break

class FixedPricingStrategy:
    def __init__(self, fixed_price: float = 2.0):
        self.fixed_price = fixed_price
        self.name = "Fixed Pricing"
        
    def select_price(self, cluster: int,  ef: int, t: int) -> float:
        return self.fixed_price
    
    def update(self, cluster:int, ef: int, price: float, demand: float):
        pass

class RandomPricingStrategy:
    def __init__(self, config: ExperimentConfig):
        self.price_range = config.price_range
        self.name = "Random Pricing"
        
    def select_price(self, cluster:int, ef: int, t: int) -> float:
        return np.random.uniform(self.price_range[0], self.price_range[1])
    
    def update(self, cluster:int, ef: int, price: float, demand: float):
        pass

class LinearDemandPricing:
    def __init__(self, config: ExperimentConfig):
        self.price_range = config.price_range
        self.models = {}
        self.history = {}
        self.name = "Linear Demand Pricing"
        self.exploration_prob = 0.1
        self.min_samples = 10
        
    def select_price(self, cluster: int,  ef: int, t: int) -> float:
        if ef not in self.history:
            self.history[ef] = {'prices': [], 'demands': []}
        
        if len(self.history[ef]['prices']) < self.min_samples:
            return np.random.uniform(self.price_range[0], self.price_range[1])
        
        if np.random.random() < self.exploration_prob:
            return np.random.uniform(self.price_range[0], self.price_range[1])
        
        if ef in self.models:
            model = self.models[ef]
            try:
                a = model.intercept_
                b = -model.coef_[0]
                if b > 0:
                    optimal_price = min(a / (2 * b), self.price_range[1])
                    optimal_price = max(optimal_price, self.price_range[0])
                    return optimal_price
            except:
                pass
        
        return np.random.uniform(self.price_range[0], self.price_range[1])
    
    def update(self, cluster:int, ef: int, price: float, demand: float):
        if ef not in self.history:
            self.history[ef] = {'prices': [], 'demands': []}
        
        self.history[ef]['prices'].append(price)
        self.history[ef]['demands'].append(demand)
        
        if len(self.history[ef]['prices']) >= 5:
            X = np.array(self.history[ef]['prices']).reshape(-1, 1)
            y = np.array(self.history[ef]['demands'])
            
            model = LinearRegression()
            model.fit(X, y)
            self.models[ef] = model

class ConvexDemandPricing:
    def __init__(self, config: ExperimentConfig):
        self.price_range = config.price_range
        self.models = {}
        self.history = {}
        self.name = "Convex Demand Pricing"
        self.exploration_prob = 0.1
        self.min_samples = 15
        
    def select_price(self, cluster: int, ef: int, t: int) -> float:
        if ef not in self.history:
            self.history[ef] = {'prices': [], 'demands': []}
        
        if len(self.history[ef]['prices']) < self.min_samples:
            return np.random.uniform(self.price_range[0], self.price_range[1])
        
        if np.random.random() < self.exploration_prob:
            return np.random.uniform(self.price_range[0], self.price_range[1])
        
        if ef in self.models:
            model = self.models[ef]
            
            prices = np.linspace(self.price_range[0], self.price_range[1], 50)
            best_revenue = -np.inf
            best_price = prices[0]
            
            for p in prices:
                X = np.array([[p, p**2]])
                pred_demand = np.clip(model.predict(X)[0], 0, 1)
                revenue = p * pred_demand
                if revenue > best_revenue:
                    best_revenue = revenue
                    best_price = p
            
            return best_price
        
        return np.random.uniform(self.price_range[0], self.price_range[1])
    
    def update(self, cluster:int, ef: int, price: float, demand: float):
        if ef not in self.history:
            self.history[ef] = {'prices': [], 'demands': []}
        
        self.history[ef]['prices'].append(price)
        self.history[ef]['demands'].append(demand)
        
        if len(self.history[ef]['prices']) >= 10:
            prices = np.array(self.history[ef]['prices'])
            X = np.column_stack([prices, prices**2])
            y = np.array(self.history[ef]['demands'])
            
            model = Ridge(alpha=1.0)
            model.fit(X, y)
            self.models[ef] = model