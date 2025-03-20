import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importance(model, X_train):
    feature_importances = model.get_feature_importance()
    feature_names = X_train.columns

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    importance_df = importance_df.head(15)

    plt.figure(figsize=(10, 5))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance - CatBoost")
    plt.gca().invert_yaxis()  # Pour afficher la plus importante en haut
    plt.show()

    return importance_df

def shap_analysis(X_test, model):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)

def test_each_feature_importance(X_train, X_test, y_train,y_test):
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
