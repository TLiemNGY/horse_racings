import pandas as pd

def preprocess_data(df):
    df = pd.get_dummies(df, columns=['venue', 'config', 'race_class', 'surface', 'distance', 'going', 'horse_country',
                                     'horse_type'], dtype=int, drop_first=True)

    df['place_odds'].fillna(0, inplace=True)
    return df

def split_train_test(df, train_ratio = 0.8):
    unique_races = df.index.unique()

    split_index = int(len(unique_races) * train_ratio)

    train_races = unique_races[:split_index]
    test_races = unique_races[split_index:]

    df_train = df[df.index.isin(train_races)]
    df_test = df[df.index.isin(test_races)]

    y_train = df_train['won']
    X_train = df_train.drop(columns=['won', 'horse_id'])

    y_test = df_test['won']
    X_test = df_test.drop(columns=['won', 'horse_id'])
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