import pandas as pd
import numpy as np

def make_ts_features(df):
    
    if ('item_id' in df.columns):
        df = df.drop('item_id', axis=1)

    df["date_num"] = df.index.astype('int') / 10**9
    df["date_month"] = df.index.month
    df['date_month_lbl'] = df.index.month_name() 
    df['date_month_lbl'] = pd.Categorical(
        df['date_month_lbl'],
        categories=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', ' November', 'December']
    )


    df['date_wday'] = df.index.dayofweek + 1
    df['date_wday_lbl'] = df.index.day_name() 
    df['date_wday_lbl'] = pd.Categorical(
        df['date_wday_lbl'],
        categories=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    )
    df['weekend'] = np.where(df.index.dayofweek <= 5, 0, 1)

    df = pd.get_dummies(df)

    return df