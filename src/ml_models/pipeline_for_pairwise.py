import pandas as pd
from lightgbm import LGBMClassifier
from data_processing.utils import load_data, define_cat_features, fill_na, split_train_test, create_prediction_column, convert_features_for_ranker
from data_processing.merge_data import merge_data
from data_processing.clean_data import clean_raw_data, clean_before_fit
from data_processing.feature_engineering import add_interaction_features, add_combined_cat_features, add_lag_features, simplify_features
from data_processing.pairwise_approach_utils import generate_pairwise_dataset, split_pairwise_by_race
from metrics.utils_for_metrics import predict_pairwise_winner

def run_pairwise_pipeline(model_name, use_feature_engineering=True, use_bet_odds=True):
    # Load data
    df_runs = load_data("../data/raw/runs.csv")
    df_races = load_data("../data/raw/races.csv")


    # Merge & clean
    df_runs, df_races = clean_raw_data(df_runs, df_races)
    df = merge_data(df_runs, df_races)
    df = create_prediction_column(df)

    df['race_id'] = df['race_id'].astype(int)


    df, cat_features = define_cat_features(df, model_name)
    if use_feature_engineering:
        df = add_lag_features(df)
        df = add_interaction_features(df)
        df = simplify_features(df)
        df, cat_features = add_combined_cat_features(df, cat_features, model_name)
        df = convert_features_for_ranker(df, cat_features)

    df = clean_before_fit(df, model_name)
    df = fill_na(df, model_name)

    if not use_bet_odds:
        df.drop(columns=['win_odds', 'place_odds'], inplace=True)

    # Créer pairwise dataset
    feature_cols = [col for col in df.columns if col not in ['horse_id']]
    #print(df.info())
    df_pairwise = generate_pairwise_dataset(df, target_col='result', feature_cols=feature_cols)
    print(df_pairwise)

    # Split races proprement
    df_train, df_test = split_pairwise_by_race(df_pairwise, train_ratio=0.8)
    df_test_original = df_test.copy()

    return df_test, df_train, df_test_original

    X_train = df_train.drop(columns=['race_id', 'horse_a_id', 'horse_b_id', 'target','result','won'])
    y_train = df_train['target']

    X_test = df_test.drop(columns=['race_id', 'horse_a_id', 'horse_b_id', 'target','result','won'])
    y_test = df_test['target']

    model = LGBMClassifier(objective='binary', metric='binary_logloss')
    model.fit(X_train, y_train)

    # df_test_races est le df original (non pairwise) avec toutes les features
    df_preds = predict_pairwise_winner(df_test_races=df_test_original, model=model, feature_cols=feature_cols)
    accuracy = (df_preds['predicted_winner'] == df_preds['true_winner']).mean() * 100
    print(f"✅ Précision gagnant pairwise : {accuracy:.2f}%")

    return model, df_pairwise, df_test
