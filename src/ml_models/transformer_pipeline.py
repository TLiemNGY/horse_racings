import pandas as pd
import torch

from data_processing.clean_data import (clean_before_fitting, clean_data_races, clean_data_runs)
from data_processing.feature_engineering import (add_lag_features, feature_engineering, add_interaction_features)
from data_processing.load_data import load_data
from data_processing.merge_data import merge_data
from data_processing.output_to_csv import output_to_csv
from data_processing.preprocess_data import (fill_na, preprocess_data, create_prediction_column, split_train_test)
from metrics.performance_metrics import (compute_accuracy, compute_top_prediction_transformer, finalize_dataframe, prepare_predictions)
from ml_models.transformer_model import train_transformer, predict_transformer

def run_transformer_model(model_name):
    # Load datasets
    df_runs = load_data("../data/raw/runs.csv")
    df_races = load_data("../data/raw/races.csv")

    # Clean data
    clean_data_runs(df_runs)
    clean_data_races(df_races)
    df = merge_data(df_runs, df_races)
    df = create_prediction_column(df)
    df = add_lag_features(df)  # inactif pour l'instant
    clean_before_fitting(df, model_name)

    # Preprocessing data regarding model
    df = fill_na(df)
    df = preprocess_data(df, model_name)
    df, cat_features = feature_engineering(df, model_name)
    df = add_interaction_features(df)

    # Séparation Train/Test
    y_train, y_test, X_train, X_test, df_train, df_test = split_train_test(df, train_ratio=0.8, relative_or_binary_ranking='result')
    df_test_original = df_test.copy()

    print(X_train.info())

    # --- Entraînement du modèle Transformer ---
    model, encoders, scaler = train_transformer(df_train)
    torch.save(model.state_dict(), "../models/transformer_model.pth")
    print("✅ Modèle Transformer entraîné et sauvegardé.")

    # --- Prédictions sur le Test Set ---
    y_predict = predict_transformer(df_test, model, encoders, scaler)

    # Adapter les prédictions au DataFrame original
    df_pred = prepare_predictions(df_test_original, y_predict)

    # Identifier le cheval avec la prédiction la plus haute
    df_pred = compute_top_prediction_transformer(df_pred)

    # Finaliser les statistiques
    df_stats = finalize_dataframe(df_pred)
    accuracy = compute_accuracy(df_stats)

    # Sauvegarde des résultats
    output_to_csv(df_stats, 'transformer_stats')
    print(f"✅ Précision du modèle Transformer : {accuracy:.2f}%")

    return model, df_stats
