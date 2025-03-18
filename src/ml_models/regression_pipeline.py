import pandas as pd
from catboost import CatBoostRanker, CatBoostRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from data_processing.clean_data import (clean_before_fitting, clean_data_races,
                                        clean_data_runs)
from data_processing.feature_engineering import (add_lag_features,
                                                 feature_engineering)
from data_processing.load_data import load_data
from data_processing.merge_data import merge_data
from data_processing.output_to_csv import output_to_csv
from data_processing.preprocess_data import (fill_na, preprocess_data,
                                             split_train_test, create_prediction_column)
from metrics.performance_metrics import (compute_accuracy,
                                         compute_top_prediction,
                                         finalize_dataframe,
                                         prepare_predictions)
from ml_models.linear_regression import train_linear_regression


def run_model(model_name, use_feature_engineering, use_bet_odds=True):
    # Load datasets
    df_runs = load_data("../data/raw/runs.csv")
    df_races = load_data("../data/raw/races.csv")

    # Clean data
    clean_data_runs(df_runs)
    clean_data_races(df_races)
    df = merge_data(df_runs, df_races)
    df = create_prediction_column(df)
    df = add_lag_features(df)
    clean_before_fitting(df, model_name)

    # Preprocessing data regarding model
    df = fill_na(df)
    df = preprocess_data(df, model_name)
    df.set_index("race_id", inplace=True)

    # Feature engineering
    if use_feature_engineering:
        df, cat_features = feature_engineering(df, model_name)

    else :
        cat_features = ['venue', 'config', 'race_class', 'surface', 'distance', 'going', 'horse_country',
                               'horse_type', 'horse_gear', 'trainer_id', 'jockey_id', 'horse_rating','horse_ratings','track_conditions']

    if not use_bet_odds:
        print(df.columns)
        df.drop(columns=['win_odds', 'place_odds'], inplace=True)


    y_train, y_test, X_train, X_test, df_train, df_test = split_train_test(df, train_ratio=0.8, relative_or_binary_ranking='relative_ranking')
    df_test_original = df_test.copy()

    # Print infos au besoin
    print(X_train.info())
    #print(y_train.head(14))
    #print(X_test.info())
    #print(X_train.columns)


    # Choix du modèle
    if model_name == "linear_regression":
        print('training linear regression model')
        model = LinearRegression()
        model.fit(X_train, y_train)


    elif model_name == "xgboost":
        print ('training xgboost model')
        model = XGBRegressor(enable_categorical=True)
        model.fit(X_train, y_train)


    elif model_name == "catboost":
        print ('training catboost model')

        model = CatBoostRegressor(
            cat_features=cat_features,
            iterations=1000,
            learning_rate=0.05,
            loss_function= 'RMSE',
            depth=8,
            verbose=100)
        model.fit(X_train, y_train)

    else:
        raise ValueError(f"invalid model_name: {model_name}. choose from ['linear_regression', 'xgboost', 'catboost'.")


    # Faire des prédictions sur X_test
    y_predict = model.predict(X_test)

    if model_name in ['catboost','xgboost','linear_regression']:
        # Prédiction et évaluation
        df = prepare_predictions(df_test_original, y_predict)
        df = compute_top_prediction(df)
        df_stats = finalize_dataframe(df)

        accuracy = compute_accuracy(df_stats)

        # Output to csv for analysis
        output_to_csv(df_stats, 'statistiques')

        print(f"✅ Précision basée sur les courses : {accuracy:.2f}%")

    return model, X_train, df_stats
