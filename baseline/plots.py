import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# METHOD_LABELS, METHOD_COLORS, _to_rgba and plot_trajectory_plotly generated using Claude (opus-5.5 medium)


#readable names and one fixed colour per method, so a method looks the same in every plot
METHOD_LABELS = {'rs': 'Random search', 'desc': 'Descriptors', 'desc+prior': 'Descriptors + prior',
                 'onehot': 'One-hot', 'onehot+prior': 'One-hot + prior'}
METHOD_COLORS = {'Random search': '#6b6a66', 'Descriptors': '#2a78d6', 'Descriptors + prior': '#eb6834',
                 'One-hot': '#1baf7a', 'One-hot + prior': '#eda100'}


def _to_rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    return f'rgba({r},{g},{b},{alpha})'


def plot_trajectory_plotly(res: pd.DataFrame, r1: str, r2: str, n_init: int=10, methods: list=None, budget: int=None) -> go.Figure:
    """
    Interactive best-yield-so-far curves for one reactant pair.
    res: long benchmark table (method, r1, r2, iter, seed, best, max_start, max_yield),
         i.e. the outputs of benchmark() and benchmark_rs() stacked with pd.concat.
    methods: method names as stored in res (e.g. ['rs', 'onehot']); None shows all.
    budget: optional total number of experiments to mark with a vertical line.
    Line = median over seeds, band = 25th to 75th percentile over seeds.
    The x-axis counts all experiments, so the first point is the start set (n_init).
    """
    pair = res[(res['r1']==r1) & (res['r2']==r2)]
    if methods is not None:
        pair = pair[pair['method'].isin(methods)]
    if pair.empty:
        raise ValueError(f'No benchmark results for {r1} + {r2}')

    #best of the start points, one value per seed, placed at x = n_init
    start = pair[pair['iter']==pair['iter'].min()].copy()
    start['iter'] = 0
    start['best'] = start['max_start']
    curves = pd.concat([start, pair], ignore_index=True)
    curves['experiments'] = curves['iter'] + n_init
    curves['label'] = curves['method'].map(METHOD_LABELS).fillna(curves['method'])

    stats = (curves.groupby(['label', 'experiments'])['best']
             .quantile([0.25, 0.5, 0.75]).unstack().reset_index()
             .rename(columns={0.25: 'q25', 0.5: 'median', 0.75: 'q75'}))
    order = [label for label in METHOD_COLORS if label in set(stats['label'])]
    order += [label for label in stats['label'].unique() if label not in order]

    lines = px.line(stats, x='experiments', y='median', color='label', custom_data=['q25', 'q75'],
                    category_orders={'label': order}, color_discrete_map=METHOD_COLORS)
    lines.update_traces(line_width=2, hovertemplate='%{y:.1f}%  (middle half of seeds: %{customdata[0]:.1f} to %{customdata[1]:.1f})')
    lines.update_traces(selector=dict(name='Random search'), line_dash='dash')

    #bands go in first so they sit behind the lines; same legendgroup = they hide together with their line
    bands = []
    for trace in lines.data:
        method_stats = stats[stats['label']==trace.name]
        hidden = dict(mode='lines', line_width=0, legendgroup=trace.legendgroup, showlegend=False, hoverinfo='skip')
        bands.append(go.Scatter(x=method_stats['experiments'], y=method_stats['q75'], **hidden))
        bands.append(go.Scatter(x=method_stats['experiments'], y=method_stats['q25'], fill='tonexty',
                                fillcolor=_to_rgba(trace.line.color, 0.15), **hidden))
    fig = go.Figure(data=bands + list(lines.data), layout=lines.layout)

    pair_best = pair['max_yield'].iloc[0]
    fig.add_hline(y=pair_best, line_dash='dot', line_width=1, line_color='#8a8984',
                  annotation_text=f'Best in dataset: {pair_best:.1f}%', annotation_position='top left')
    #headroom above the best yield so the annotation does not sit on the curves
    low = stats['q25'].min()
    fig.update_yaxes(range=[low - 0.03*(pair_best - low), pair_best + 0.12*(pair_best - low)])
    if budget is not None:
        fig.add_vline(x=budget, line_dash='dot', line_width=1, line_color='#8a8984')

    fig.update_layout(title=f'{r1} + {r2}', hovermode='x unified', legend_title_text='', template='plotly_white',
                      xaxis_title='Experiments run (first point = start set)', yaxis_title='Best yield found (%)',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                      margin=dict(l=10, r=10, t=90, b=10))
    fig.update_xaxes(dtick=5)
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