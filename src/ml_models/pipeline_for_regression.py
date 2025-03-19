import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression

from data_processing.merge_data import merge_data
from data_processing.feature_engineering import add_interaction_features, add_combined_cat_features, add_lag_features
from data_processing.utils import load_data, define_cat_features, output_data_to_csv, create_prediction_column, fill_na, split_train_test
from metrics.metrics import compute_global_accuracy
from metrics.utils_for_metrics import add_predictions_to_df_test, build_output_dataframe, compute_top_prediction_per_race
from xgboost import XGBRegressor
from data_processing.clean_data import clean_raw_data, clean_before_fit

def run_preprocess_pipeline(model_name, use_feature_engineering, use_bet_odds):
    # Load datasets
    df_runs = load_data("../data/raw/runs.csv")
    df_races = load_data("../data/raw/races.csv")

    # Clean data
    df_runs, df_races = clean_raw_data(df_runs, df_races)
    df = merge_data(df_runs, df_races)
    df = create_prediction_column(df)

    # Add features
    df, cat_features = define_cat_features(df, model_name)

    if use_feature_engineering:
        df = add_lag_features(df)
        df = add_interaction_features(df)
        df, cat_features = add_combined_cat_features(df, cat_features, model_name)

    # Last adjustments before fit
    df = clean_before_fit(df, model_name)
    df = fill_na(df)

    # Should we use bet odds to predict results ?
    if not use_bet_odds:
        df.drop(columns=['win_odds', 'place_odds'], inplace=True)


    y_train, y_test, X_train, X_test, df_train, df_test = split_train_test(df, choice='relative_ranking', train_ratio=0.8)

    # Print infos au besoin
    print(X_train.info())
    print(y_train)
    #print(X_test.info())
    #print(y_test)
    return y_train, y_test, X_train, X_test, df_train, df_test, cat_features


def run_model(model_name, use_feature_engineering, use_bet_odds=True):
    # Run la pipeline de préparation de la data
    global df_stats
    y_train, y_test, X_train, X_test, df_train, df_test, cat_features = run_preprocess_pipeline(model_name, use_feature_engineering, use_bet_odds)

    # Choix du modèle
    if model_name == "linear_regression":
        print('training linear regression model')
        model = LinearRegression()
        model.fit(X_train, y_train)

    elif model_name == "xgboost":
        print ('training xgboost model')
        model = XGBRegressor(objective="reg:squarederror", eval_metric="rmse",enable_categorical=True)
        model.fit(X_train, y_train)

    elif model_name == "catboost":
        print ('training catboost model')

        model = CatBoostRegressor(loss_function= 'RMSE', cat_features=cat_features)
        model.fit(X_train, y_train)

    else:
        raise ValueError(f"invalid model_name: {model_name}. choose from ['linear_regression', 'xgboost', 'catboost'].")

    # Faire des prédictions sur X_test
    y_predict = model.predict(X_test)

    if model_name in ['catboost','xgboost','linear_regression']:
        # Prédiction et évaluation
        df = add_predictions_to_df_test(df_test, y_predict)

        df = df.reset_index() # A supprimer quand on aura reglé ce pb...
        df = compute_top_prediction_per_race(df)
        df_stats = build_output_dataframe(df)

        accuracy = compute_global_accuracy(df_stats)

        # Output to csv for analysis
        output_data_to_csv(df_stats, 'statistiques')

        print(f"✅ Précision basée sur les courses : {accuracy:.2f}%")

    return model, X_train, df_stats
