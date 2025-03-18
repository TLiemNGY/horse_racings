import pandas as pd


def add_lag_features(df):
    #df['date'] = pd.to_datetime(df['date'])
    #df['mean_result_last_5'] = df.groupby('horse_id')['result'].shift().transform(lambda x: x.rolling(3).mean()).fillna(0)
    #df['win_rate_last_5'] = df.groupby('horse_id')['won'].shift().transform(lambda x: x.rolling(5).mean()).fillna(0)
    #df['avg_lengths_behind_last_3'] = df.groupby('horse_id')['lengths_behind'].shift().transform(lambda x: x.rolling(3).mean()).fillna(0)
    #df['rank_change_last_3'] = df.groupby('horse_id')['result'].shift().diff(3).fillna(0)
    #df['days_since_last_race'] = df.groupby('horse_id')['date'].diff().dt.days.shift().fillna(0)
    return df

def feature_engineering(df, model_name):
    cat_features = ['venue', 'config', 'race_class', 'surface', 'distance', 'going', 'horse_country',
                    'horse_type', 'horse_gear', 'trainer_id', 'jockey_id', 'horse_rating', 'horse_ratings']

    # Variables sans prétraitement
    df['global_weight'] = df['actual_weight'] - df['declared_weight']

    if model_name == 'linear_regression':
        df = df

    elif model_name in ['xgboost','catboost']:
        #df['track_conditions'] = df['going'].astype(str) + "_" + df['surface'].astype(str) + "_" + df['config'].astype(str)
        #df['track_conditions'] = df['track_conditions'].astype("category")
        #cat_features.append('track_conditions')

        df['specialized_trainer'] = df['trainer_id'].astype(str) + "_" + df['jockey_id'].astype(str) + "_" + df['horse_rating'].astype(str)
        df['specialized_trainer'] = df['specialized_trainer'].astype("category")
        cat_features.append('specialized_trainer')

        #df['distance_vs_age'] = df['distance'].astype(str) + "_" + df['horse_age'].astype(str)
        #df['distance_vs_age'].astype('category')
        #cat_features.append('distance_vs_age')

        #df['country_specificity'] = df['horse_country'].astype(str) + "_" + df['horse_type'].astype(str) + "_" + df['horse_ratings'].astype(str)
        #df['country_specificity'].astype('category')
        #cat_features.append('country_specificity')

        #df['distance_vs_weight'] = df['distance'].astype(str) + "_" + df['actual_weight'].astype(str)
        #df['distance_vs_weight'].astype('category')
        #cat_features.append('distance_vs_weight')

        # df['weather'] = df['mean_temp'] + df['going']

        df = df

    return df, cat_features

# Rajouter la météo