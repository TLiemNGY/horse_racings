import pandas as pd

def prepare_predictions(df_test_original, y_predict):
    df = df_test_original.copy()
    df['y_predict'] = y_predict
    return df


def compute_top_prediction(df):
    df['max_prediction'] = df.groupby(level=0)['y_predict'].transform('max')
    df['top_prediction'] = (df['y_predict'] == df['max_prediction']).astype(int)
    df.drop(columns=['max_prediction'], inplace=True)
    return df


def finalize_dataframe(df):
    df = df[['horse_id', 'won', 'y_predict', 'top_prediction']]
    df['won'] = df['won'].astype(int)
    return df


def compute_accuracy(df):
    df['accuracy'] = ((df['won'] == 1) & (df['top_prediction'] == 1))

    total_races = df.index.nunique()
    correct_prediction_races = df.groupby(df.index)['accuracy'].max().sum()

    accuracy = (correct_prediction_races / total_races) * 100
    return accuracy