import pandas as pd

def compute_accuracy(df_test, y_predict):
    df_test['y_predict'] = y_predict

    df_test['top_prediction'] = (df_test['y_predict'] == df_test.groupby(level=0)['y_predict'].transform('max')).astype(int)
    df_test['won'] = df_test['won'].astype(int)
    return df_test[['horse_id', 'won', 'y_predict', 'top_prediction']]