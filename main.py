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
import warnings
warnings.filterwarnings('ignore')
from experiment.config import ExperimentConfig
from experiment.experiment_runner import ExperimentRunner
from dataset.data_loader import load_data

def main():
    config = ExperimentConfig()
    data, queries = load_data(config, './dataset/base.fvecs', './dataset/query.fvecs')
    runner = ExperimentRunner(config, data=data, query=queries)
    results = runner.run_comparison_experiment(T=10000, demand_type='bimodal')
    print(results)

if __name__ == '__main__':
    main()
