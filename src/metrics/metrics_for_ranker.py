import pandas as pd

def prepare_ranker_predictions(df_test_original, y_predict):
    df = df_test_original.copy()
    df['y_predict'] = y_predict
    return df

def compute_ranking(df):
    df['rank'] = df.groupby(level=0)['y_predict'].rank(ascending=False, method='first')
    return df

def finalize_ranker_dataframe(df):
    df = df[['horse_id', 'won', 'y_predict', 'rank']]
    df['won'] = df['won'].astype(int)
    return df

def compute_top1_accuracy(df):
    df['is_correct'] = (df['won'] == 1) & (df['rank'] == 1)

    total_races = df.index.nunique()
    correct_prediction_races = df.groupby(df.index)['is_correct'].max().sum()

    accuracy = (correct_prediction_races / total_races) * 100
    return accuracy
