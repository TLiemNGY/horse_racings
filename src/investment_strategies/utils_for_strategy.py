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
