import pandas as pd

def clean_raw_data(df_runs, df_races):
    df_runs.drop(columns=['finish_time', 'horse_no',
                     'position_sec1', 'position_sec2', 'position_sec3', 'position_sec4', 'position_sec5',
                     'position_sec6',
                     'behind_sec1', 'behind_sec2', 'behind_sec3', 'behind_sec4', 'behind_sec5', 'behind_sec6',
                     'time1', 'time2', 'time3', 'time4', 'time5', 'time6'], inplace=True)

    df_races.drop(columns=['sec_time1', 'sec_time2', 'sec_time3', 'sec_time4', 'sec_time5', 'sec_time6', 'sec_time7',
                     'time1', 'time2', 'time3', 'time4', 'time5', 'time6', 'time7'
                     ], inplace=True)
    return df_runs, df_races
    # Ajouter la vérification des colonnes : c'est ici que je dois clean horse_gear, etc.


def clean_before_fit(df, model_name):
    if model_name == "linear_regression":

        df.drop(columns=['date','race_no','prize',
                          'place_combination1','place_combination2','place_combination3','place_combination4',
                          'place_dividend1','place_dividend2','place_dividend3','place_dividend4',
                          'win_combination1','win_dividend1','win_combination2','win_dividend2',
                          'lengths_behind','horse_gear','trainer_id','jockey_id','horse_rating','horse_ratings'], inplace=True)

    elif model_name in ['xgboost','catboost','transformer','lgbm']:

        df.drop(columns=['date','race_no','prize',
                          'place_combination1','place_combination2','place_combination3','place_combination4',
                          'place_dividend1','place_dividend2','place_dividend3','place_dividend4',
                          'win_combination1','win_dividend1','win_combination2','win_dividend2',
                          'lengths_behind'], inplace=True)

    return df