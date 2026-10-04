import pandas as pd
import streamlit as st

#reads HTE csv file and cleans it 
@st.cache_data
def load_data(path="data/suzuki.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path, index_col='Reaction_No')
    column_mapper = {'Reactant_1_Name':'reactant1', 'Reactant_2_Name':'reactant2',
                     'Ligand_Short_Hand':'ligand', 'Reagent_1_Short_Hand':'reagent',
                     'Solvent_1_Short_Hand':'solvent', 'Product_Yield_PCT_Area_UV':'yield_uv'}
    df = df[column_mapper.keys()]
    df.rename(columns=column_mapper, inplace=True)
    di = {'MeOH/H2O_V2 9:1':'MeOH', 'THF_V2':'THF'}
    df.replace({'solvent':di}, inplace=True)
    df.fillna({'ligand':'No_Ligand', 'reagent':'No_Reagent'}, inplace = True)
    return df              

#filters the dataset for one reactant pair
def filter_pair(df: pd.DataFrame, pair: tuple, one_hot=False) -> pd.DataFrame:
    if one_hot:
        filtered = df[df[pair[0]] & df[pair[1]]]
        filtered.drop([pair[0], pair[1]], axis=1, inplace=True)
        features = ['yield_uv', 'AmPhos', 'CataCXium A', 'No_Ligand',
       'P(Cy)3', 'P(Ph)3', 'P(o-Tol)3', 'P(tBu)3', 'SPhos', 'XPhos',
       'Xantphos', 'dppf', 'dtbpf', 'CsF', 'Et3N', 'K3PO4', 'KOH', 'LiOtBu',
       'NaHCO3', 'NaOH', 'No_Reagent', 'DMF', 'MeCN', 'MeOH', 'THF']
        return filtered[features]
    filtered = df[(df['reactant1']==pair[0]) & (df['reactant2']==pair[1])].copy()
    str_columns = ['reactant1', 'reactant2', 'ligand', 'reagent', 'solvent']
    filtered.drop(str_columns, axis = 1, inplace=True)
    return filtered

#return one-hot encoded df
def get_one_hot(X: pd.DataFrame) -> pd.DataFrame:
    oh = pd.get_dummies(X)
    oh.columns = [column_name.split("_")[-1] for column_name in oh.columns]
    col_mapper = {"Ligand":"No_Ligand", "Reagent":"No_Reagent"}
    oh.rename(columns=col_mapper, inplace=True)
    return oh

#returns available pairs for streamlit
@st.cache_data
def list_pairs(df: pd.DataFrame) -> tuple:
    rct_1 = [name for name in df['reactant1'].unique()]
    rct_2 = [name for name in df['reactant2'].unique()]
    return (rct_1, rct_2)

