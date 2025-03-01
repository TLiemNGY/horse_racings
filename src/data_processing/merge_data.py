import pandas as pd

def merge_data(df_runs, df_races)
    df = pd.merge(df_races, df_runs, on='race_id', how='inner')
    return df