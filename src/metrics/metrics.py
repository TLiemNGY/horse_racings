def compute_global_accuracy(df):
    df['accuracy'] = ((df['won'] == 1) & (df['top_prediction'] == 1))

    total_races = df.index.nunique()
    correct_prediction_races = df.groupby(df.index)['accuracy'].max().sum()

    accuracy = (correct_prediction_races / total_races) * 100
    return accuracy
