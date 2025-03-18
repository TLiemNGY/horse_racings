import pandas as pd


def kelly_criterion(df, amount_bet, duration, min_kelly, calibrate_probabilities=True, single_bet_per_race=True):
    df = df.head(duration).copy()

    if calibrate_probabilities:
        df['y_predict_calibrated'] = df.groupby('race_id')['y_predict'].transform(lambda x: x / x.sum())
    else:
        df['y_predict_calibrated'] = df['y_predict']

    # Calcul des probabilités implicites des cotes
    df['theoretical_probability'] = 1 / df['win_odds']

    # Calcul du facteur Kelly
    df['kelly_fraction'] = ((df['y_predict_calibrated'] * (df['win_odds'] - 1)) - (1 - df['y_predict_calibrated'])) / (df['win_odds'] - 1)

    # Ne parier que si la fraction Kelly est positive et dépasse un seuil
    df['kelly_fraction'] = df['kelly_fraction'].apply(lambda x: max(x, 0))  # Pas de mise négative

    # Sélectionner un seul pari par course (si activé)
    if single_bet_per_race:
        df['max_kelly_per_race'] = df.groupby('race_id')['kelly_fraction'].transform('max')
        df = df[df['kelly_fraction'] == df['max_kelly_per_race']]

    # Appliquer le filtre min_kelly
    df.loc[df['kelly_fraction'] < min_kelly, 'kelly_fraction'] = 0

    # Calcul des mises en fonction du bankroll
    df['bet_amount'] = df['kelly_fraction'] * amount_bet

    # Calcul des gains
    df['gains'] = df.apply(lambda row: row['bet_amount'] * row['win_dividend1'] if row['won'] == 1
    else -row['bet_amount'], axis=1)

    return df

