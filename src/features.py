import json
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_descriptors(path="data/descriptors.json"):
    with open(path) as f:
        return json.load(f)

def build_features(df, descriptors):
    #look up each row's solvent/base/ligand in the dicts, concatenate
    idx = df.index
    for name, desc_dict in descriptors.items():
        df_dict = pd.DataFrame(desc_dict).transpose()
        df = df.merge(df_dict, how='left', left_on=name, right_index=True)
    df.set_index(idx, inplace=True)
    desc_cols = ["polarity_solvent", "dielectric_solvent", "bp_solvent", "viscosity_solvent",
             "mw_reagent", "pKa_reagent", "solubility_h2o", "mw_ligand", "logP_ligand"]
    df[desc_cols] = df[desc_cols].apply(pd.to_numeric, errors="raise")
    df.drop(['smiles_solvent', 'smiles_reagent', 'smiles_ligand',
              'solubility_methanol', 'solubility_ethanol', 'rotatable_bonds'], axis=1, inplace=True)
    return df


def var_scaler(df: pd.DataFrame) -> pd.DataFrame:
    #normalizes numerical variables for GP
    scaler = StandardScaler()
    num_columns = df.select_dtypes(include='float').columns
    y = df['yield_uv'].copy()
    df[num_columns] = scaler.fit_transform(df[num_columns].astype(float))
    df['yield_uv'] = y
    return df

def prior_yields(df: pd.DataFrame, filtered: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ligand, reagent, solvent), data in df.groupby(["ligand", "reagent", "solvent"]):
        intersection_idx = data.index.difference(filtered.index)
        mean_yield = data["yield_uv"].loc[intersection_idx].mean()
        for idx in data.index:
            rows.append(dict(prior_yield=mean_yield, Reaction_No = idx))
    result = pd.DataFrame(rows)
    result.set_index("Reaction_No", inplace=True)
    return result
