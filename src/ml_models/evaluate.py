import pandas as pd
from ml_models.pipeline import run_model
from data_processing.preprocess_data import fetch_winning_dividends
from investment_strategies.baseline_strategy import baseline_strategy
from metrics.performance_metrics import compute_accuracy


def compare_models(model_configs):

    results = []

    for model_name, use_feature_engineering, use_bet_odds in model_configs:
        df_stats = run_model(model_name, use_feature_engineering, use_bet_odds)

        # Récupérer les gains après 1500 courses
        df = fetch_winning_dividends(df_stats)
        df = baseline_strategy(df, 100, 1500)

        # Calcul du gain final après 1500 courses
        final_gain = df['cumulative_gain'].iloc[-1]

        # Calcul de l'accuracy
        accuracy = compute_accuracy(df_stats)

        # Calcul du ROI (Retour sur investissement)
        initial_investment = 100 * 1500  # Mise de 100 unités sur 1500 courses
        roi = (final_gain - initial_investment) / initial_investment * 100  # En pourcentage

        results.append({
            'Modèle': model_name,
            'Feature Engineering': use_feature_engineering,
            'Utilisation des Cotes': use_bet_odds,
            'Accuracy (%)': round(accuracy, 2),
            'Gain final': round(final_gain, 2),
            'ROI (%)': round(roi, 2)
        })

    df_results = pd.DataFrame(results)

    # Affichage des résultats
    import ace_tools as tools
    tools.display_dataframe_to_user(name="Comparaison des Modèles", dataframe=df_results)

    return df_results
