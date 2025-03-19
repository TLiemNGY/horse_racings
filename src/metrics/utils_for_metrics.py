import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

def add_predictions_to_df_test(df_test_before_training, y_predict):
    df = df_test_before_training.copy()
    df['y_predict'] = y_predict
    return df

def compute_top_prediction_per_race(df):
    df['max_prediction'] = df.groupby('race_id')['y_predict'].transform('max')
    df['top_prediction'] = (df['y_predict'] == df['max_prediction']).astype(int)
    df.drop(columns=['max_prediction'], inplace=True)
    return df

def build_output_dataframe(df):
    df = df[['race_id','horse_id', 'won', 'y_predict', 'top_prediction']]
    df['won'] = df['won'].astype(int)
    return df
    
