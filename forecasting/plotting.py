
import pandas as pd
import numpy as np
import plotly.express as px


def plot_forecasts(data, facet_ncol = 1, id_slice = None):

    df = data.copy()

    ids = df['item_id'].unique()

    if id_slice is not None:
        ids = ids[id_slice]

        _filter = df['item_id'].isin(ids)

        df = df[_filter]

    ret = df \
        .melt(
            id_vars    = ['index', 'item_id'],
            value_vars = ['value', 'pred_ets', 'pred_xgb'], 
            value_name = 'val'
        ) \
        .pipe(
            func           = px.line,
            x              = 'index', 
            y              = 'val',
            color          = 'variable', 
            facet_col      = "item_id", 
            facet_col_wrap = facet_ncol,
            render_mode    = 'svg',
            line_shape     = 'spline',
            color_discrete_sequence = ["#0055AA", "#C40003", "#00C19B"],
            template       = "plotly_dark"
        ) \
        .update_yaxes(matches=None) \
        .update_layout(showlegend=True, font = dict(size=8)) \
        .update_traces(line = dict(width=0.7))

    return ret