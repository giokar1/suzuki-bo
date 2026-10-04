from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel, ConstantKernel as C
import numpy as np

def fit_gp(X, y, n_restarts_optimizer=5, seed=42):
    #builds a GP with a Matern Kernel
    kernel = (C(1.0, (1e-2, 1e2))
              * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 1e1), nu=2.5) 
              + WhiteKernel(noise_level=0.04, noise_level_bounds="fixed"))
    gp = GaussianProcessRegressor(kernel=kernel, random_state=seed, normalize_y=True, n_restarts_optimizer=n_restarts_optimizer)
    gp.fit(X, y)
    return gp

def predict(gp, X):
    #returns the predicted mean and std
    pred, std = gp.predict(X, return_std=True)
    return (pred, std)


def fit_rf_baseline(X, y):
    #fits an RF
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf
