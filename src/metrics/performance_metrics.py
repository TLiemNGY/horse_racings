import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

def prepare_predictions(df_test_original, y_predict):
    df = df_test_original.copy()
    df['y_predict'] = y_predict
    return df


def compute_top_prediction(df):
    df['max_prediction'] = df.groupby(level=0)['y_predict'].transform('max')
    df['top_prediction'] = (df['y_predict'] == df['max_prediction']).astype(int)
    df.drop(columns=['max_prediction'], inplace=True)
    return df

def compute_top_prediction_transformer(df):
    df['max_prediction'] = df.groupby('race_id')['y_predict'].transform('max')
    df['top_prediction'] = (df['y_predict'] == df['max_prediction']).astype(int)
    df.drop(columns=['max_prediction'], inplace=True)
    return df

def finalize_dataframe(df):
    df = df[['horse_id', 'won', 'y_predict', 'top_prediction']]
    df['won'] = df['won'].astype(int)
    return df


def compute_accuracy(df):
    df['accuracy'] = ((df['won'] == 1) & (df['top_prediction'] == 1))

    total_races = df.index.nunique()
    correct_prediction_races = df.groupby(df.index)['accuracy'].max().sum()

    accuracy = (correct_prediction_races / total_races) * 100
    return accuracy


def shap_analysis(X_test, model):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)


def test_features(X_train, X_test, y_train,y_test):
    model = LinearRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    mae_before = mean_absolute_error(y_test,y_pred)

    # Store results
    results = []

    # Test sur toutes les features
    for feature in X_train.columns:
        X_train_reduced = X_train.drop(columns=[feature])
        X_test_reduced = X_test.drop(columns=[feature])

        # Model retraining
        model.fit(X_train_reduced,y_train)
        y_pred_reduced = model.predict(X_test_reduced)
        mae_after = mean_absolute_error(y_test,y_pred_reduced)
        delta = mae_after-mae_before

        # Add results to storage
        results.append({"Feature": feature, "MAE Before": mae_before, "MAE after": mae_after, "Delta": delta})

    results = pd.DataFrame(results)

    # Trier les résultats par Delta (impact de la feature)
    results = results.sort_values(by="Delta", ascending=False)

    # Affichage du top 10 des features les plus influentes
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=results.head(10),
        x="Delta",
        y="Feature",
        palette="coolwarm"
    )
    plt.xlabel("Variation de MAE (Delta)")
    plt.ylabel("Feature")
    plt.title("Classement des features qui dégradent le plus la précision du modèle")
    plt.axvline(0, color='black', linestyle='dashed', linewidth=1)
    plt.show()

    return results

    
