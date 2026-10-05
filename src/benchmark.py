import pandas as pd
from src.data import get_one_hot
from src.features import prior_yields
from src.optimize import run_bo, run_random_search


# max_iter does not include starting points
def benchmark(df, n_seeds=10, n_init=10, max_iter=30, cutoff_pos=5, cutoff_frac=0.95, one_hot = False, prior_values=False):
    rows = []
    descriptors = [
        "polarity_solvent", "dielectric_solvent", "bp_solvent", "viscosity_solvent",
        "mw_reagent", "pKa_reagent", "solubility_h2o", "mw_ligand", "logP_ligand"
    ]
    if one_hot:
        label="onehot"
    else:
        label="desc"

    if prior_values:
        label += "+prior"
    
    for (r1, r2), filtered in df.groupby(['reactant1', 'reactant2']):
        if one_hot:
            one_hot_block = filtered[["ligand", "solvent", "reagent"]].copy()
            one_hot_block = get_one_hot(one_hot_block)
            X = one_hot_block
        else:
            descriptor_block = filtered[descriptors].copy()
            X = descriptor_block
        if prior_values:
            prior_block = prior_yields(df, filtered).loc[filtered.index]
            X = X.merge(prior_block, on="Reaction_No")
        X.index = filtered.index
        y = filtered["yield_uv"].copy()

        cutoff_by_pos = sorted(y, reverse=True)[cutoff_pos-1]
        if len(y) < max_iter + n_init:
            #skipping pairs with no experiments
            continue
        max_yield = filtered["yield_uv"].max()
        for seed in range(n_seeds):
            bo, max_start = run_bo(X, y, max_iter, n_init=n_init, init_seed=seed, prior_values=prior_values)
            for iter in range(max_iter):
                rows.append(dict(method=label, r1=r1, r2=r2, iter=iter+1, seed=seed+1, best=bo[iter], max_start = max_start, max_yield=max_yield, cutoff_by_pos = float(bo[iter] >= cutoff_by_pos), cutoff_by_frac=float(bo[iter]/max_yield >= cutoff_frac)))
    result = pd.DataFrame(rows)
    return result


def benchmark_rs(df, n_seeds=10, n_init=10, max_iter=30, rs_runs=100, cutoff_pos=5, cutoff_frac=0.95):
    rows = []
    for (r1, r2), filtered in df.groupby(['reactant1', 'reactant2']):
        y=filtered["yield_uv"].copy()
        max_yield = y.max()
        cutoff_by_pos_thr = sorted(filtered["yield_uv"], reverse=True)[cutoff_pos-1]

        for seed in range(n_seeds):
            rs = run_random_search(y, max_iter, rs_runs, n_init, init_seed=seed, seed=seed)
            max_start = y.sample(n=n_init, random_state=seed).max()
            for iter in range(max_iter):
                #counting runs that finished with yield within threshold of the maximum
                cutoff_by_frac = sum(rs[:, iter]/max_yield >= cutoff_frac)/rs_runs
                #counting runs that finished within top {cutoff_pos} of best candidates
                cutoff_by_pos = sum(rs[:, iter]>= cutoff_by_pos_thr)/rs_runs
                rows.append(dict(method="rs", r1=r1, r2=r2, iter=iter+1, seed=seed+1, best=rs[:,iter].mean(), max_start = max_start, max_yield=max_yield, cutoff_by_pos=cutoff_by_pos, cutoff_by_frac=cutoff_by_frac))
    res = pd.DataFrame(rows)
    return res


def get_metrics(res: pd.DataFrame, exp_num: int, get_table=False) -> pd.DataFrame:
    #exp_num does not include initially picked experiments
    rows = []
    for (method, r1, r2), method_res in res.groupby(["method", "r1", "r2"]):
        thr_by_pos = method_res[(method_res["iter"]==exp_num)]["cutoff_by_pos"].mean()
        thr_by_frac = method_res[(method_res["iter"]==exp_num)]["cutoff_by_frac"].mean()
        rows.append(dict(method=method, r1=r1, r2=r2, thr_by_pos=thr_by_pos, thr_by_frac=thr_by_frac))
    metrics = pd.DataFrame(rows)
    table_rows = []
    if get_table:
        for method, method_df in metrics.groupby("method"):
            table_rows.append(dict(method=method, thr_by_pos=method_df["thr_by_pos"].mean(), thr_by_frac=method_df["thr_by_frac"].mean()))
        return pd.DataFrame(table_rows)
    return metrics