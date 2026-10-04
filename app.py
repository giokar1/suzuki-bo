import warnings
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data import load_data, filter_pair, list_pairs, get_one_hot
from src.features import load_descriptors, build_features, var_scaler, prior_yields
from src.optimize import expected_improvement, run_bo, run_random_search
from src.model import predict, fit_gp, fit_rf_baseline
from baseline.plots import plot_feature_importance, plot_trajectory, plot_tsne
from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings("ignore", category=ConvergenceWarning)
# -------------------------------------------------------


path_data = 'data/suzuki.xlsx'
path_desc = 'data/descriptors.json'
df = load_data(path_data)
descriptors = load_descriptors(path_desc)
df_desc = build_features(df, descriptors)
df_desc = var_scaler(df_desc)
# one-hot encoding for importance plot
# df_one_hot = get_one_hot(df)


# -------------------------------------------------------
st.set_page_config(
    page_title="Suzuki BO",
    page_icon="⚗️",
    layout="wide"
)


st.title("Suzuki Coupling Optimizer")
st.write("Bayesian optimization for reaction condition selection using the AstraZeneca HTE dataset.")

left, right = st.columns(2)
with left:
    reactant1_choice = st.selectbox("Select 1st Reactant:", list_pairs(df)[0])
with right:
    if reactant1_choice in ['6-quinoline-boronic acid hydrochloride', 'Potassium quinoline-6-trifluoroborate', '6-Quinolineboronic acid pinacol ester']:
        reactant2_choice = st.selectbox("Second Reactant:", list_pairs(df)[1][-1])
    else:
        reactant2_choice = st.selectbox("Select 2nd Reactant:", list_pairs(df)[1][:-1])



filtered = filter_pair(df, (reactant1_choice, reactant2_choice))

y = X['yield_uv'].copy()
X = X.merge(prior_yields(df, X).loc[X.index], on="Reaction_No")
X

X.drop('yield_uv', axis=1, inplace=True)
init_seed = 42
bo_hist, _ = run_bo(X, y, 30, n_init=10, init_seed=init_seed, prior_values=True)
rs = run_random_search(y, 30, runs =100, n_init=10, init_seed=init_seed)
best_theoretical = np.max(y)
fig = plot_trajectory(bo_hist, rs.mean(axis=0), rs.std(axis=0), best_theoretical)
st.pyplot(fig)