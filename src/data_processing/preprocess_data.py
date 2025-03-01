import pandas as pd

def preprocess_data(df):
    df = pd.get_dummies(df, columns=['venue', 'config', 'race_class', 'surface', 'distance', 'going', 'horse_country',
                                     'horse_type'], dtype=int, drop_first=True)

    df['place_odds'].fillna(0, inplace=True)
    return df

def split_train_test(df, train_ratio = 0.8):
    split_index = int(len(df) * train_ratio)

    df_train = df.iloc[:split_index]
    df_test = df.iloc[split_index:]

    y_train = df_train['won']
    X_train = df_train.drop(columns=['won', 'horse_id'])

    y_test = df_test['won']
    X_test = df_test.drop(columns=['won', 'horse_id'])
    return y_train, y_test, X_train, X_test, df_train, df_test