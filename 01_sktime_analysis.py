# BUSINESS SCIENCE LEARNING LABS ----
# LAB 64: SKTIME FORECASTING ----
# **** ----

# CONDA ENV USED: lab_64_sktime (see environment.yml for instructions)

# LIBRARIES & DATA ----

import pandas as pd
import numpy as np
import plotly.express as px

df = pd.read_csv("data/walmart_item_sales.csv")
df

df.info()

# Reformat Data Types

df['item_id'] = pd.Categorical(df['item_id'])

df['date'] = pd.to_datetime(df['date'])

df.info()

# Visualize

_filter = df['item_id'].isin(df['item_id'].unique()[:12])
_filter = df['item_id'].isin(df['item_id'].unique()[-12:])

df_filtered = df[_filter]

df_filtered \
    .pipe(
        func           = px.line,
        x              = 'date', 
        y              = 'value', 
        color          = 'item_id',
        facet_col      = "item_id", 
        facet_col_wrap = 3,
        render_mode    = 'svg',
        line_shape     = 'spline',
        template       = "plotly_dark"
    ) \
    .update_yaxes(matches=None) \
    .update_layout(showlegend=False, font = dict(size=8)) \
    .update_traces(line = dict(width=0.7))

# TIP 1: TRY 1 TIME SERIES -----

# Select Time Series ----
n = 0

_filter = df['item_id'].isin(
    # Make sure it's a list
    [df['item_id'].unique()[n]]
)

df_sample = df[_filter] \
    .set_index('date') 
df_sample

df_sample.index.freq = 'd'
df_sample.index

df_sample['value'] = df_sample['value'].astype('float')
df_sample.info()

# Split the time series ----
from sktime.forecasting.model_selection import temporal_train_test_split

df_train, df_test = temporal_train_test_split(
    df_sample, 
    test_size=90
)

df_train.index

df_test.index



# ETS MODEL ----

# Forecast Horizon ----
fh = np.arange(1, 91)
fh

# Forecasting - ETS Model

from sktime.forecasting.ets import AutoETS

forecaster_ets = AutoETS(
    auto=True, 
    njobs=-1, 
    sp = 7, 
    additive_only=True
)
forecaster_ets

forecaster_ets.fit(y = df_train['value'])

y_pred_ets = forecaster_ets.predict(fh)
y_pred_ets.name = 'pred_ets'
y_pred_ets

# Visualize Results - ETS Model

results_df = pd.concat([df_sample, y_pred_ets], axis=1) \
    .reset_index() 
    
results_df \
    .pipe(
        func      = px.line,
        x         = 'index', 
        y         = ['value', 'pred_ets'], 
        template  = "plotly_dark", 
        render_mode    = 'svg',
        line_shape     = 'spline',
    )

# Performance - ETS Model

import sklearn.metrics as metrics
import sktime.performance_metrics.forecasting as ts_metrics

assess_df = pd.concat([df_test, y_pred_ets], axis=1)

metrics.r2_score(
    y_true=assess_df['value'], 
    y_pred=assess_df['pred_ets']
)

ts_metrics.mean_absolute_error(
    y_true=assess_df['value'], 
    y_pred=assess_df['pred_ets']
)

# Future Forecast - ETS Model ----

forecaster_ets.update(
    y = df_sample['value']
)

future_pred_ets = forecaster_ets.predict(fh) \
    .rename("pred_ets")

# XGBOOST MODEL ----

df_sample

# Feature Engineering -----

from forecasting.ts_features import make_ts_features

df_sample_features = make_ts_features(df_sample)

df_train, df_test = temporal_train_test_split(
    df_sample_features, 
    test_size=90
)

# Xgboost Model ----

from xgboost import XGBRegressor

forecaster_xgb = XGBRegressor()

forecaster_xgb.fit(
    X = df_train.drop('value', axis=1),
    y = df_train['value']
)

# Performance - XGBoost Model ----

y_pred_xgb = forecaster_xgb.predict(df_test.drop('value', axis=1))

y_pred_xgb = pd.Series(y_pred_xgb, index=df_test.index) \
    .rename("pred_xgb")

assess_df = pd.concat([df_test.value, y_pred_xgb], axis=1) \
    .reset_index()

metrics.r2_score(
    y_true=assess_df['value'], 
    y_pred=assess_df['pred_xgb']
)

ts_metrics.mean_absolute_error(
    y_true=assess_df['value'], 
    y_pred=assess_df['pred_xgb']
)

# FUTURE FORECAST ----

future_series = pd.date_range(
    start = df_sample_features.index[-1] + pd.DateOffset(days=1),
    periods = 90
)

X_future = pd.DataFrame(dict(date = future_series)) \
    .set_index('date') \
    .pipe(
        make_ts_features
    )

forecaster_xgb.fit(df_sample_features.drop('value', axis=1), df_sample_features['value'])

future_pred_xgb = forecaster_xgb.predict(X_future)

future_pred_xgb = pd.Series(future_pred_xgb, index=future_series) \
    .rename("pred_xgb")

future_pred_xgb

# FUTURE VISUALIZATION - ETS & XGBOOST ----

pd.concat([df_sample, future_pred_ets, future_pred_xgb], axis = 1) \
    .assign(item_id = lambda x: x['item_id'].ffill()) \
    .reset_index() \
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
        render_mode    = 'svg',
        line_shape     = 'spline',
        color_discrete_sequence = ["#0055AA", "#C40003", "#00C19B"],
        template       = "plotly_dark"
    ) \
    .update_layout(showlegend=True, font = dict(size=8)) \
    .update_traces(line = dict(width=0.7))

# 2.0 SCALE ----
# - LL PRO MEMBERS GET THE CODE
# - Tip #2: Handling Errors
# - Tip #3: Review errors & reforecast using different methods

from forecasting.at_scale import run_forecasts
from forecasting.plotting import plot_forecasts

df = pd.read_csv("data/walmart_item_sales.csv")
df

df.info()

# Run Automation

best_forecasts_df = run_forecasts(
    df, 
    id_slice=None
)

best_forecasts_df.to_pickle("data/best_forecasts_df.pkl")

best_forecasts_df = pd.read_pickle("data/best_forecasts_df.pkl")

# Plot Automation

plot_forecasts(
    best_forecasts_df, 
    facet_ncol = 3, 
    id_slice = np.arange(0,12)
)




