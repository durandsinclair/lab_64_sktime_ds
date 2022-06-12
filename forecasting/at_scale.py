

import logging
import os

import pandas as pd
import numpy as np
import plotly.express as px

from sktime.forecasting.model_selection import temporal_train_test_split

import sklearn.metrics as metrics
import sktime.performance_metrics.forecasting as ts_metrics

from xgboost import XGBRegressor
from sktime.forecasting.ets import AutoETS

from forecasting.ts_features import make_ts_features


def run_forecasts(data, verbose=True, log_errors=True, id_slice=None):

    error_file = "error_logs.txt"
    if log_errors:
        if os.path.exists(error_file):
            os.remove(error_file)

    df = data.copy()

    # Prep Data
    df['item_id'] = pd.Categorical(df['item_id'])
    df['date'] = pd.to_datetime(df['date']) 

    # Make iterables
    ids = df['item_id'].unique()
    
    if id_slice is not None:
        ids = ids[id_slice]

    n_obs = len(ids)

    li = []

    for i, id in enumerate(ids):

        try:
            
            if verbose: print(f"[{i+1}/{n_obs}] {id}")

            # Select Single Time Series
            df_sample = select_time_series(df, n=i)

            # Split Time Series ---- 
            df_train, df_test = temporal_train_test_split(
                df_sample, 
                test_size=90
            )

            # ETS Model ----
            ets_results = ets_train_test(df_train, df_test)

            if verbose:
                print(f"  [SUCCESS] ETS Model R-squared: {np.round(ets_results['score'],3)}")

            # XGBoost Model ----
            xgb_results = xgb_train_test(df_train, df_test)

            if verbose:
                print(f"  [SUCCESS] XGB Model R-squared: {np.round(xgb_results['score'],3)}")

            # Compare ----
            if (ets_results['score'] > xgb_results['score']):

                if verbose:
                    print(f"  [MODEL SELECTED] ETS")

                forecaster_ets = ets_results['forecaster']

                forecaster_ets.update(
                    y = df_sample['value']
                )

                n_obs = df_test.shape[0]
                fh = np.arange(1, n_obs+1)

                future_pred = forecaster_ets.predict(fh) \
                    .rename("pred_ets")

            else:
                if verbose:
                    print(f"  [MODEL SELECTED] XGBOOST")

                forecaster_xgb = xgb_results['forecaster']

                df_sample_features = make_ts_features(df_sample)

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

                future_pred = pd.Series(future_pred_xgb, index=future_series) \
                    .rename("pred_xgb")
            
            

            # Collect Results

            predictions_df = pd.concat([df_sample, future_pred], axis = 1) \
                .assign(item_id = lambda x: x['item_id'].ffill()) \
                .reset_index()
            
            # print(predictions_df)

            li = li + [predictions_df]

        except Exception as Argument:
            
            msg = f"An error occurred in {i}: {id}."

            if verbose:
                logging.exception(msg)

            if log_errors:
                f = open(error_file, "a")
                f.write(msg)
                f.write("\n")
                f.write(str(Argument))
                f.write("\n\n")
                f.close()
            
        print(id)
    
    ret = pd.concat(li, axis=0)

    return ret


# FUNCTIONS ----

def select_time_series(data, n = 0):

    df = data.copy()

    # Select Time Series ----
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

    return df_sample



def ets_train_test(df_train, df_test):

    # ETS MODEL ----

    # Forecast Horizon ----
    n_obs = df_test.shape[0]
    fh = np.arange(1, n_obs+1)

    # Forecasting - ETS Model
    forecaster_ets = AutoETS(
        auto=True, 
        njobs=-1, 
        sp = 7, 
        additive_only=True
    )

    forecaster_ets.fit(y = df_train['value'])

    y_pred_ets = forecaster_ets.predict(fh)
    y_pred_ets.name = 'pred_ets'
    y_pred_ets 

    # Metrics
    assess_df = pd.concat([df_test, y_pred_ets], axis=1)

    score = metrics.r2_score(
        y_true=assess_df['value'], 
        y_pred=assess_df['pred_ets']
    )

    ret = dict(
        score = score,
        forecaster = forecaster_ets
    )

    return ret

def xgb_train_test(df_train, df_test):
    
    # XGBOOST MODEL ----

    # Feature Engineering
    df_train = make_ts_features(df_train)
    df_test = make_ts_features(df_test)

    # Xgboost Model 
    forecaster_xgb = XGBRegressor()

    forecaster_xgb.fit(
        X = df_train.drop('value', axis=1),
        y = df_train['value']
    )

    # Performance 
    y_pred_xgb = forecaster_xgb.predict(df_test.drop('value', axis=1))

    y_pred_xgb = pd.Series(y_pred_xgb, index=df_test.index) \
        .rename("pred_xgb")

    assess_df = pd.concat([df_test.value, y_pred_xgb], axis=1) \
        .reset_index()

    score = metrics.r2_score(
        y_true=assess_df['value'], 
        y_pred=assess_df['pred_xgb']
    )

    # Return 
    ret = dict(
        score = score,
        forecaster = forecaster_xgb
    )

    return ret

