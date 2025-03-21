import pandas as pd
import itertools

def generate_pairwise_dataset(df, target_col='result', feature_cols=None):
    pairwise_rows = []

    for race_id, group in df.groupby(df.index):  # index = race_id
        horses = group.to_dict('records')
        for a, b in itertools.combinations(horses, 2):
            row = {}

            # Identifiants
            row['race_id'] = race_id
            row['horse_a_id'] = a['horse_id']
            row['horse_b_id'] = b['horse_id']

            # Label : 1 si A finit devant B, 0 sinon
            row['target'] = 1 if a[target_col] < b[target_col] else 0

            # Différences de features
            for col in feature_cols:
                row[f'diff_{col}'] = a[col] - b[col]

            pairwise_rows.append(row)

    return pd.DataFrame(pairwise_rows)

def split_pairwise_by_race(df_pairwise, train_ratio):
    unique_races = df_pairwise['race_id'].unique()
    split_index = int(len(unique_races) * train_ratio)

    train_races = unique_races[:split_index]
    test_races = unique_races[split_index:]

    df_train = df_pairwise[df_pairwise['race_id'].isin(train_races)]
    df_test = df_pairwise[df_pairwise['race_id'].isin(test_races)]

    return df_train, df_test
