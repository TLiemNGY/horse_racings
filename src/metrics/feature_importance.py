import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importance(model, X_train):
    feature_importances = model.get_feature_importance()
    feature_names = X_train.columns

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 5))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance - CatBoost")
    plt.gca().invert_yaxis()  # Pour afficher la plus importante en haut
    plt.show()

    return importance_df
