import pandas as pd

from data_processing.output_to_csv import output_to_csv


def baseline_strategy(df, amount_bet, duration):
    df = df.head(duration).copy()

    df['gains'] = df.apply(lambda row: -amount_bet if not row['accuracy']
                           else amount_bet*row['win_dividend1'], axis=1)

    output_to_csv(df,"baseline_strategy")
    return df