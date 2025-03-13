import matplotlib.pyplot as plt

def plot_cumulative_gains(df, bets, strategy_name="Baseline Strategy"):
    df['cumulative_gains'] = df['gains'].cumsum()

    plt.figure(figsize=(10, 5))
    plt.plot(df['race_id'], df['cumulative_gains'], marker='o', linestyle='-')
    plt.xlabel("Race ID")
    plt.ylabel("Cumulative Gains (HKD)")
    plt.title(f"Cumulative Gains for {strategy_name} with {bets} bets")
    plt.grid(True)
    plt.show()

    return df

def plot_cumulative_gains_per_horse(df, bets, strategy_name="Baseline Strategy"):
    df['cumulative_gains'] = df['gains'].cumsum()

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['cumulative_gains'], marker='o', linestyle='-')
    plt.xlabel("Index")
    plt.ylabel("Cumulative Gains (HKD)")
    plt.title(f"Cumulative Gains for {strategy_name} with {bets} bets")
    plt.grid(True)
    plt.show()