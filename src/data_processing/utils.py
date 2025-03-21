import os
import pandas as pd
import numpy as np

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def output_data_to_csv(df, filename):
    df = df.reset_index()
    save_path = os.path.join("..", "data", "processed", f"{filename}.csv")
    df.to_csv(save_path, index=False)

def create_prediction_column(df):
    df['total_horses'] = df.groupby('race_id')['result'].transform('max')
    alpha = 2  # Facteur d'amplification des pénalités
    df['relative_ranking'] = 1 - np.tanh(alpha * ((df['result'] - 1) / (df['total_horses'] - 1)))
    return df

def add_target_choices(df_train, df_test, choice):

    if choice=='won':
        y_train = df_train['won']
        y_test = df_test['won']

    elif choice=='relative_ranking':
        y_train = df_train['relative_ranking']
        y_test = df_test['relative_ranking']

    elif choice=='result':
        y_train = df_train['result']
        y_test = df_test['result']

    X_train = df_train.drop(columns=['won','result','relative_ranking'])
    X_test = df_test.drop(columns=['won','result','relative_ranking'])

    return y_train, y_test, X_train, X_test


def split_train_test(df, choice, train_ratio=0.8):
    unique_races = df['race_id'].unique()
    df.set_index("race_id", inplace=True)

    split_index = int(len(unique_races) * train_ratio)

    train_races = unique_races[:split_index]
    test_races = unique_races[split_index:]

    df_train = df[df.index.isin(train_races)]
    df_test = df[df.index.isin(test_races)]

    y_train, y_test, X_train, X_test = add_target_choices(df_train, df_test, choice)

    return y_train, y_test, X_train, X_test, df_train, df_test

def fill_na(df, model_name):
    if model_name =='lgbm':
        df['place_odds'].fillna(df['place_odds'].mean(), inplace=True)

    else:
        df['place_odds'].fillna(df['place_odds'].mean(), inplace=True)
        df['horse_country'] = df['horse_country'].cat.add_categories("Unknown").fillna("Unknown")
        df['horse_type'] = df['horse_type'].cat.add_categories("Unknown").fillna("Unknown")
        df['horse_type'] = df['horse_type'].fillna('Unknown')

    return df


def define_cat_features(df, model_name):
    if model_name == "linear_regression":
        cat_features = df[
            ['venue', 'config', 'race_class', 'surface', 'distance', 'going', 'horse_country', 'horse_type']]
        df = pd.get_dummies(df, columns=cat_features, dtype=int, drop_first=True)

    else:
        cat_features = ['venue', 'config', 'race_class', 'surface', 'distance', 'going', 'horse_country',
                        'horse_type', 'trainer_id', 'jockey_id', 'horse_rating', 'horse_ratings']

        for col in cat_features:
            df[col] = df[col].astype("category")

    return df, cat_features


def convert_features_for_ranker(df, cat_features, one_hot_threshold=10):
    """
    - One-hot encode les features catégorielles avec peu de modalités (≤ threshold)
    - Frequency encode les autres
    """
    nunique_dict = {col: df[col].nunique() for col in cat_features}
    print(nunique_dict)

    one_hot_cols = [col for col, n in nunique_dict.items() if n <= one_hot_threshold]
    freq_enc_cols = [col for col in cat_features if col not in one_hot_cols]

    # One-hot encoding
    df = pd.get_dummies(df, columns=one_hot_cols, dtype=int, drop_first=True)

    # Frequency encoding
    for col in freq_enc_cols:
        df[f'{col}_freq'] = df[col].map(df[col].value_counts())

    return df