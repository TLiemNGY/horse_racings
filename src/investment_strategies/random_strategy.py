import numpy as np


def random_strategy(df, amount_bet, duration):
    df = df.head(duration).copy()

    random_value = np.random.rand(len(df))
    df['gains'] = np.where(random_value < (1/df['num_horses']),
                           amount_bet * df['win_dividend1'],
                           -amount_bet)
    return df