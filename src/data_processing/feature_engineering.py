import pandas as pd


def add_lag_features(df):
    df['date'] = pd.to_datetime(df['date'])

    df['mean_result_last_5'] = df.groupby('horse_id')['result'].shift().transform(lambda x: x.rolling(3).mean()).fillna(0) # OK
    df['win_rate_last_10'] = df.groupby('horse_id')['won'].shift().transform(lambda x: x.rolling(10).mean()).fillna(0) # OK
    df['avg_lengths_behind_last_5'] = df.groupby('horse_id')['lengths_behind'].shift().transform(lambda x: x.rolling(5).mean()).fillna(0) # OK
    df['rank_change_last_3'] = df.groupby('horse_id')['result'].shift().diff(3).fillna(0) # OK
    df['rank_change_last_10'] = df.groupby('horse_id')['result'].shift().diff(10).fillna(0) # OK
    df['days_since_last_race'] = df.groupby('horse_id')['date'].diff().dt.days.shift().fillna(0) # Hmm
    return df

def add_interaction_features(df):
    df["race_avg_win_odds"] = df.groupby("race_id")["win_odds"].transform("mean")
    df["race_std_win_odds"] = df.groupby("race_id")["win_odds"].transform("std")
    df["relative_odds"] = df["win_odds"] / df["race_avg_win_odds"]

    df["race_avg_weight"] = df.groupby("race_id")["actual_weight"].transform("mean")
    df["relative_weight"] = df["actual_weight"] / df["race_avg_weight"]
    df['global_weight'] = df['actual_weight'] - df['declared_weight']

    df["relative_draw"] = df["draw"] / df["total_horses"]

    df.drop(columns=['declared_weight','race_avg_weight'], inplace=True)
    return df

def add_combined_cat_features(df, cat_features, model_name):

    if model_name in ['xgboost','catboost','transformer','lgbm']:
        df['track_conditions'] = df['going'].astype(str) + "_" + df['surface'].astype(str) + "_" + df['config'].astype(str)
        df['track_conditions'] = df['track_conditions'].astype("category")
        cat_features.append('track_conditions')

        df['specialized_trainer'] = df['trainer_id'].astype(str) + "_" + df['jockey_id'].astype(str) + "_" + df['horse_rating'].astype(str)
        df['specialized_trainer'] = df['specialized_trainer'].astype("category")
        cat_features.append('specialized_trainer')

        df['distance_vs_age'] = df['distance'].astype(str) + "_" + df['horse_age'].astype(str)
        df['distance_vs_age'].astype('category')
        cat_features.append('distance_vs_age')

        df['country_specificity'] = df['horse_country'].astype(str) + "_" + df['horse_type'].astype(str) + "_" + df['horse_ratings'].astype(str)
        df['country_specificity'].astype('category')
        cat_features.append('country_specificity')

        df['distance_vs_weight'] = df['distance'].astype(str) + "_" + df['actual_weight'].astype(str)
        df['distance_vs_weight'].astype('category')
        cat_features.append('distance_vs_weight')

        # df['weather'] = df['mean_temp'] + df['going']
    return df, cat_features

def simplify_features(df):
    top_gears = ["B", "TT", "H", "CP", "SR", "XB", "V", "P", "TT/B", "TT/H"]

    def transform_gear(gear):
        for g in top_gears:
            if g in gear:
                return g
        return "OTHER"

    df["horse_gear_simplified"] = df["horse_gear"].apply(transform_gear)
    df = pd.get_dummies(df, columns=["horse_gear_simplified"], prefix="gear")
    df.drop(columns=["horse_gear"], inplace=True)
    return df
