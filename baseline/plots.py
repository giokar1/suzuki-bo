import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def plot_trajectory(bo_hist: np.ndarray, rand_means: np.ndarray, rand_std: np.ndarray, best_theo: np.float64=0.0) -> plt.Figure:
    # best = float(np.max(bo_hist.to_numpy()))
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlim(0, len(bo_hist)+1)
    ax.plot(range(1, len(bo_hist)+1), bo_hist, marker = 'o', label='BO', color='r')
    ax.plot(range(1, len(rand_means)+1), rand_means, marker = 'o', label='Random Search')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Predicted Yield')
    ax.set_title('Bayesian optimization vs Random selection')
    if best_theo != 0:
        ax.axhline(y=best_theo, color='purple', linestyle='--', label=f'Maximum in the Dataset: {best_theo:.1f}%')
    ax.axhline(y=np.max(bo_hist), color='r', linestyle='--', label=f'Maximum BO: {np.max(bo_hist):.1f}%')
    ax.axhline(y=np.max(rand_means), color='b', linestyle='--', label=f'Maximum RS: {np.max(rand_means):.1f}%')
    ax.fill_between(range(1, len(rand_means)+1), y1=rand_means-rand_std, y2=rand_means+rand_std,alpha=0.4)
    
    ax.grid(True)
    ax.legend(loc='best')
    return fig

def plot_trajectory_plotly(bo_hist, rand_means, rand_std, best_theo=0):
    pass
    # best = float(np.max(bo_hist.to_numpy()))
    x = np.arange(1, len(bo_hist)+1)
    fig = px.line()
    
    return fig


def plot_feature_importance(model, feature_names):
    #Gini importance plotted
    importances = model.feature_importances_
    importances = pd.DataFrame({'Feature': feature_names , 'Gini Importance': importances}).sort_values('Gini Importance', ascending=False)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.barh(importances['Feature'][:7], importances['Gini Importance'][:7])
    ax.set_xlabel('Gini Importance')
    ax.set_title('Feature Importance - Gini Importance')
    ax.invert_yaxis()
    return fig


def plot_feature_importance_go(model, feature_names):
    importances = model.feature_importances_
    importances = pd.DataFrame({'Feature': feature_names , 'Gini Importance': importances}).sort_values('Gini Importance', ascending=False)
    fig = go.Figure(go.Bar(
        x=importances['Gini Importance'][:7],
        y=importances['Feature'][:7]
    ))
    return fig


def plot_tsne(X, y):
    #tSNE plot
    pass