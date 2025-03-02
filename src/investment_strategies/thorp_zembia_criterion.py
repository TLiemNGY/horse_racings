import numpy as np
from scipy.optimize import minimize

def thorp_zembia_criterion(df, amount_bet, duration, min_kelly=0.01):
    """
    Applique le critère de Thorp-Zembia pour optimiser les mises sur une course.

    :param df: DataFrame contenant 'race_id', 'y_predict', 'win_odds', 'won'
    :param amount_bet: Montant total à allouer par course
    :param duration: Nombre de courses prises en compte
    :param min_kelly: Seuil minimal pour prendre un pari
    :return: DataFrame avec 'bet_amount' optimisé
    """
    df = df.head(duration).copy()

    def optimize_bets(race_df):
        """ Optimisation des mises pour une seule course """
        n = len(race_df)  # Nombre de chevaux dans la course
        p = race_df['y_predict'].values  # Probabilités ML
        b = race_df['win_odds'].values - 1  # Cotes nettes

        # Fonction à maximiser : somme(p_i * log(1 + b_i * f_i))
        def objective(f):
            return -np.sum(p * np.log(1 + b * f))  # On minimise -log(Growth)

        # Contrainte : somme des mises <= 1
        constraints = ({'type': 'ineq', 'fun': lambda f: 1 - np.sum(f)})

        # Bornes : Chaque mise doit être >= 0 et <= 1
        bounds = [(0, 1) for _ in range(n)]

        # Initialisation : Répartir équitablement les mises
        f0 = np.full(n, 1 / n)

        # Optimisation
        result = minimize(objective, f0, bounds=bounds, constraints=constraints)

        if result.success:
            return result.x  # Retourne les mises optimisées
        else:
            return np.zeros(n)  # En cas d’échec, on ne mise pas

    # Appliquer l'optimisation pour chaque course
    df['bet_fraction'] = 0  # Initialisation
    for race_id, race_df in df.groupby('race_id'):
        fractions = optimize_bets(race_df)
        df.loc[race_df.index, 'bet_fraction'] = fractions

    # Appliquer un filtre min_kelly
    df.loc[df['bet_fraction'] < min_kelly, 'bet_fraction'] = 0

    # Calcul des mises en fonction du bankroll
    df['bet_amount'] = df['bet_fraction'] * amount_bet

    # Calcul des gains
    df['gains'] = df.apply(lambda row: row['bet_amount'] * row['win_dividend1'] if row['won'] == 1
                           else -row['bet_amount'], axis=1)

    return df
