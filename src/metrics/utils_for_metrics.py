import matplotlib.pyplot as plt
import pandas as pd
import itertools
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


def predict_pairwise_winner(df_test_races, model, feature_cols):
    """
    Prédit le gagnant pour chaque course à partir du modèle pairwise.
    """
    predictions = []

    for race_id, group in df_test_races.groupby('race_id'):
        horses = group.to_dict('records')
        scores = {h['horse_id']: 0 for h in horses}

        for a, b in itertools.combinations(horses, 2):
            row = {}
            for col in feature_cols:
                row[f'diff_{col}'] = a[col] - b[col]
            X_pair = pd.DataFrame([row])
            proba = model.predict_proba(X_pair)[0][1]
            if proba >= 0.5:
                scores[a['horse_id']] += 1
            else:
                scores[b['horse_id']] += 1

        predicted_winner = max(scores.items(), key=lambda x: x[1])[0]
        true_winner = group[group['won'] == 1]['horse_id'].values[0]

        predictions.append({
            'race_id': race_id,
            'predicted_winner': predicted_winner,
            'true_winner': true_winner
        })

    return pd.DataFrame(predictions)
