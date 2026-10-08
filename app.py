import streamlit as st
from src.data import load_data, load_results, filter_pair

from src.benchmark import get_metrics
from baseline.plots import plot_trajectory_plotly

# -------------------------------------------------------


path_data = 'data/suzuki.xlsx'
path_desc = 'data/descriptors.json'
path_results = 'data/res.csv'
df = load_data(path_data)
res = load_results(path=path_results)

col_mapper = {"thr_by_pos":"Top-5", "thr_by_frac":"≥95%"}
index_mapper = {'desc':'with descriptors', 'desc+prior':"with descriptors and prior yields", 'onehot':"one-hot encoded", 'onehot+prior':"one-hot encoded and prior yields", 'rs':'random selection'}
col_mapper_summary = {"thr_by_pos_x":"Top-5 20", "thr_by_frac_x":"≥95% 20",
                      "thr_by_pos_y":"Top-5 40", "thr_by_frac_y":"≥95% 40"}

# -------------------------------------------------------
st.set_page_config(
    page_title="Suzuki BO",
    page_icon="⚗️",
    layout="wide"
)

st.title("Suzuki Coupling Optimizer")
st.write("Bayesian optimization for reaction conditions selection using the Pereira et al. dataset.")

how_to_read = st.expander("How to read this chart?", on_change="rerun")
if how_to_read.open:
    with how_to_read:
        st.write("Each line shows the best result so far as experiments go. A line that rises faster means that the strategy needs fewer experiments to reach optimal reaction yield." \
        " X-axis includes the 10 start points used to train the model. Each line is the mean of the best yield found by the model across 10 seeded runs. The dotted line is the best possible yield for that pair of reactants. Bands show spread between seeded runs.")

with st.sidebar:
    st.write("Choose the reactants:")
    reactant1_choice = st.selectbox("Select 1st Reactant:", res["r1"].unique(), index=1)
    if reactant1_choice in ['6-quinoline-boronic acid hydrochloride', 'Potassium quinoline-6-trifluoroborate', '6-Quinolineboronic acid pinacol ester']:
        reactant2_choice = st.selectbox("Second Reactant:", "2d, Bromide")
    else:
        reactant2_choice = st.selectbox("Select 2nd Reactant:", res["r2"].unique()[:-1])
    st.divider()
    st.write("""Methods of optimization:  
    **one hot**: Features are one-hot encoded.  
    **descriptors**: Features are encoded with chemical descriptors (e.g. molecular weight, pKa)  
    **with prior**: Initial sets of experiments are chosen based on yields of conditions combinations.""")
    st.divider()
    methods = st.multiselect("Choose methods", ["one hot", "one hot with prior yields", "descriptors", "descriptors with prior yields"], default=["one hot", "descriptors", "one hot with prior yields"])
    methods_mapper = {"one hot":"onehot", "one hot with prior yields":"onehot+prior", "descriptors":"desc", "descriptors with prior yields":"desc+prior"}
    methods = [methods_mapper[method] for method in methods]
    methods.append("rs")

    filtered_res = res[(res["r1"]==reactant1_choice) & (res["r2"]==reactant2_choice)].copy()
    filtered_df = filter_pair(df, (reactant1_choice, reactant2_choice))
    display_bands = st.toggle("Display uncertainty bands", value=False)


max_yield = filtered_res["max_yield"].iloc[0]
left, right = st.columns([0.6, 0.4])

with right:
    exp_budget=st.radio("Choose the number of experiments:", [20, 40], index=1)
    st.divider()
    summary_table_pair = get_metrics(filtered_res, exp_num=exp_budget-10, get_table=True)
    summary_table_pair.set_index("method", inplace=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        col1.metric(label="Best yield", value=f"{max_yield:.2f}%")
    with col2:
        col2.metric(label="Median yield of 384 reactions:", value=f"{filtered_df["yield_uv"].median():.2f}%")
    with col3:
        col3.metric(label="Reaction within 5% of the best", value=f"{sum(filtered_df["yield_uv"]>=0.95*max_yield)}")
    st.caption(f"Method comparison for {reactant1_choice} and {reactant2_choice}.", text_alignment="center")
    st.caption("Top-5: Found top 5 best conditions; ≥95%: Found reaction within 5% of the maximum", text_alignment='center')
    st.dataframe(summary_table_pair.rename(columns=col_mapper, index=index_mapper).style.format("{:.0%}".format))
    best_yield_indices = filtered_df.nlargest(5, "yield_uv").index
    best_yield_df = df.loc[best_yield_indices][["ligand", "reagent", "solvent", "yield_uv"]]
    st.caption(f"What the optimiser is searching for: Top 5 best reaction conditions for {reactant1_choice} and {reactant2_choice}.", text_alignment="center")
    st.dataframe(best_yield_df.style.format({"yield_uv": '{:.1f}%'.format}), hide_index=True)

with left:
    with st.container(border=True):
        st.plotly_chart(plot_trajectory_plotly(res=res, r1=reactant1_choice, r2=reactant2_choice, n_init=10, methods=methods, budget=exp_budget, display_bands=display_bands))

st.divider()

st.header(f"On average, one-hot encoding with prior yields yielded in 40 experiments:", text_alignment="center")
summary_table_40 = get_metrics(res, exp_num=30, get_table=True)
summary_table_20 = get_metrics(res, exp_num=10, get_table=True)
summary_table_20.set_index("method", inplace=True)
summary_table_40.set_index("method", inplace=True)
summary_table_20.sort_values('thr_by_pos', ascending=False, inplace=True)
summary_table_20_idx = summary_table_20.index

summary = summary_table_20.merge(summary_table_40, on="method")
summary.index = summary_table_20_idx
summary.rename(index=index_mapper, columns=col_mapper_summary, inplace=True)
reached_top_5 = summary_table_40.loc["onehot+prior", "thr_by_pos"]
reached_95 = summary_table_40.loc["onehot+prior", "thr_by_frac"]
col1, col2 = st.columns(2)
with col1:
    col1.metric(label="Found a top 5 reaction:", value=f"{reached_top_5:.1%}", delta=f"{(reached_top_5-summary_table_40.loc["rs", "thr_by_pos"])*100:.3} percentage points", delta_color="green", delta_description="over RS")
with col2:
    col2.metric(label="Reached 95% yield of maximum:", value=f"{reached_95:.1%}", delta=f"{(reached_95-summary_table_40.loc["rs", "thr_by_frac"])*100:.3} percentage points", delta_color="green", delta_description="over RS")
st.divider()
st.caption("Comparison of methods across all pairs. Shown values are the means.", text_alignment='center')
st.caption("Top-5: Found top 5 best conditions; ≥95%: Found reaction within 5% of the maximum", text_alignment='center')
with st.container(horizontal_alignment='center'):
    st.dataframe(summary.style.format('{:.0%}'.format),width='content')


st.divider()
st.header("Summary")
st.write("Every presented method beat the random selection benchmark, while one-hot encoded with prior yields method nearly doubled the random search's rate of finding a top-5 reaction after 40 experiments." \
" The prior yields had the greatest effect at 20 experiments (61% vs 45%) but did not make a difference by 40.")
st.header("How the benchmark was done")

with st.expander("Expand", on_change="rerun", key=1):
    st.write("The benchmark was performed over 15 substrate pairs where each pair consisted of 384 reactions with different conditions." \
" For each pair, a method was tested over 10 seeds where each run started at 10 experiments and ran for 30 more." \
" Prior yields were calculated for each condition combination over 14 reactions excluding the reactant pair in question." \
" Some subtrate pairs had more candidates with higher yields than other therefore two metrics (Top-5, ≥95%) were implemented to took the differences into account." \
" The model used was a Gaussian process with expected improvement as its acquisition function.")
st.header("Limitations")
with st.expander("Expand", on_change="rerun", key=2):
    st.write("Each pair and method was run 10 times therefore the per-pair figures are rough and the differences of a few points between models" \
    " are within noise. Furthermore, the prior variants start from 5 informed and 5 random experiments, whereas random search always starts from 10 random points and thus some of the early advantage of prior variants comes from a better start." \
    " The prior assumes complete screens of 14 reactant pairs and actual labs would have less historical data." \
    " For a few problematic pairs, the prior yields method is misled and performs worse; future work could implement weights across pairs.")
st.header("Source and credits")
st.subheader("Dataset originally published in")
st.write("Damith Perera et al., A platform for automated nanomole-scale reaction screening and micromole-scale synthesis in flow. Science 359, 429-434(2018). Accessed 06.10.2026, DOI: 10.1126/science.aap9112")

st.subheader("AI-use")
st.write("The interactive plot function plot_trajectory_plotly in baseline/plots.py was generated by Claude (Anthropic) and tested by me. Claude was further used for code review, debugging advice and benchmarking design. All the remaining code was written by me")
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("Developed by: Giorgi Karenka")
with col2:
    st.page_link("https://github.com/giokar1/suzuki-bo", label="Github")
with col3:
    st.page_link("https://www.linkedin.com/in/giorgi-karenka-a64b9a43b/", label="LinkedIn")