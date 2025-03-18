import os


def output_to_csv(df, filename):
    df = df.reset_index()
    save_path = os.path.join("..", "data", "processed", f"{filename}.csv")
    df.to_csv(save_path, index=False)
