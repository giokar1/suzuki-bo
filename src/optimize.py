from scipy.stats import norm
import numpy as np
import pandas as pd
from src.model import fit_gp, predict
from src.data import get_one_hot

def expected_improvement(best_y, mu, sigma):
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - best_y) / sigma
    ei = (mu-best_y)*norm.cdf(z) + sigma*norm.pdf(z)
    return ei


def run_bo(X_pool: pd.DataFrame, y_pool: pd.DataFrame, max_iter: int, n_init=10, init_seed=42, prior_values=False, one_hot=False):
    n_iter = 0
    X_pool = X_pool.copy()
    y_pool = y_pool.copy()
    if prior_values:
        best_prior_yields = X_pool["prior_yield"].sort_values(ascending=False).iloc[:n_init//2]
        X_pool.drop("prior_yield", axis=1, inplace=True)
        X_tested = X_pool.loc[best_prior_yields.index]
        X_tested = pd.concat([X_tested, X_pool.drop(best_prior_yields.index).sample(n_init-n_init//2, random_state=init_seed)])
        X_pool.drop(X_tested.index, inplace=True)
    else:
        X_tested = X_pool.sample(n=n_init, random_state=init_seed)
        X_pool.drop(X_tested.index, inplace=True)

    y_tested = pd.DataFrame(y_pool.loc[X_tested.index])
    start_values = y_tested.copy()

    y_pool.drop(y_tested.index, inplace=True)

    best_yield = []
    while n_iter < max_iter:
        best_y = np.max(y_tested)
        gp = fit_gp(X_tested, y_tested, n_restarts_optimizer=10)
        mu, sigma = predict(gp, X_pool)
        scores = expected_improvement(best_y, mu, sigma)
        idx = X_pool.index[[np.argmax(scores)]]
        best_X_pool = X_pool.loc[idx]
        X_tested = pd.concat([X_tested, best_X_pool])
        y_tested = pd.concat([y_tested, y_pool.loc[idx]])
        X_pool.drop(best_X_pool.index, inplace=True)
        y_pool.drop(best_X_pool.index, inplace=True)
        best_yield.append(np.max(y_tested))
        n_iter+=1

    return np.array(best_yield), np.max(start_values)

def run_random_search(y_pool, max_iter, runs=100, n_init=10, init_seed=42, seed=0):
    rng = np.random.default_rng(seed)
    init = y_pool.sample(n=n_init, random_state=init_seed)
    best_rs_matrix = np.zeros((runs, max_iter))
    for run in range(runs):
        pool = y_pool.drop(init.index)
        y_tested = init.copy()
        for n_iter in range(max_iter):
            chosen_exp = pool.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1)))
            y_tested = pd.concat([y_tested, chosen_exp])
            pool = pool.drop(chosen_exp.index)
            best_rs_matrix[run, n_iter] = y_tested.max()

    return best_rs_matrix