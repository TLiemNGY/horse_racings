import pandas as pd
import numpy as np

def merge_data(df_runs, df_races):
    df = pd.merge(df_races, df_runs, on='race_id', how='inner')

    #df['date'] = pd.to_datetime(df['date'])
    #df = add_meteo(df)
    return df


def add_meteo(df):
    meteo = pd.read_csv("../data/raw/weather.csv")
    meteo['date'] = pd.to_datetime(meteo['date'])
    meteo['Grass_Min_Temp'] = pd.to_numeric(meteo['Grass_Min_Temp'], errors='coerce').fillna(meteo['Grass_Min_Temp'].replace('***', np.nan).astype(float).mean())
    meteo['rainfall'] = pd.to_numeric(meteo['rainfall'], errors='coerce').fillna(meteo['rainfall'].replace('Trace', np.nan).astype(float).mean())

    meteo = meteo[['date', 'mean_temp']]
    df = pd.merge(df, meteo, on='date', how='left')
    return df