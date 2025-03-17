import pandas as pd

def fill_na(df):
    df['place_odds'].fillna(df['place_odds'].mean(), inplace=True)

    df['horse_country'] = df['horse_country'].fillna('Unknown')
    df['horse_type'] = df['horse_type'].fillna('Unknown')
    return df

def preprocess_data(df, model_name):

    if model_name == "linear_regression":
        categorical_columns = ['venue', 'config', 'race_class', 'surface', 'distance', 'going', 'horse_country',
                               'horse_type']
        df = pd.get_dummies(df, columns=categorical_columns, dtype=int, drop_first=True)

    elif model_name in ["xgboost","catboost"]:
        categorical_columns = ['venue', 'config', 'race_class', 'surface', 'distance', 'going', 'horse_country',
                               'horse_type', 'horse_gear', 'trainer_id', 'jockey_id', 'horse_rating','horse_ratings']
        for col in categorical_columns:
            df[col] = df[col].astype("category")

    return df

def split_train_test(df, result_or_won, train_ratio = 0.8):
    unique_races = df.index.unique()

    split_index = int(len(unique_races) * train_ratio)

    train_races = unique_races[:split_index]
    test_races = unique_races[split_index:]

    df_train = df[df.index.isin(train_races)]
    df_test = df[df.index.isin(test_races)]

    if result_or_won=='won':
        y_train = df_train['won']
        y_test = df_test['won']

    elif result_or_won=='result':
        y_train = df_train['result']
        y_test = df_test['result']

    X_train = df_train.drop(columns=['won','result'])
    X_test = df_test.drop(columns=['won','result'])

    return y_train, y_test, X_train, X_test, df_train, df_test

def fetch_winning_dividends(df):
    df_races = pd.read_csv("../data/raw/races.csv")

    df_join = df.merge(df_races[['race_id','win_dividend1']], on=['race_id'], how='left')

    df_join['num_horses'] = df_join.groupby('race_id')['horse_id'].transform('count')
    df_join['win_dividend1'] = df_join['win_dividend1'] / 10  # 10 HKD en minimal et une unité c'est 10 HKD

    df_filtered = df_join[df_join['top_prediction'] == 1]
    return df_filtered

def fetch_winning_dividends_per_prediction(df):
    """Useful for Kelly criterion that needs all the data because you can bet on multiple horses per race"""
    df_runs = pd.read_csv("../data/raw/runs.csv")
    df_races = pd.read_csv("../data/raw/races.csv")

    df = df.merge(df_races[['race_id','win_dividend1']], on=['race_id'], how='left')
    df = df.merge(df_runs[['race_id','horse_id','win_odds']], on=['race_id', 'horse_id'], how='left')
    df.drop(columns=['top_prediction','accuracy'],inplace=True)

    df['num_horses'] = df.groupby('race_id')['horse_id'].transform('count')
    df['win_dividend1'] = df['win_dividend1'] / 10  # 10 HKD en minimal et une unité c'est 10 HKD
    return df