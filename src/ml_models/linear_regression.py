from sklearn.linear_model import LinearRegression
import joblib
import os

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Enregistrer le modèle dans "models/" à la racine
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))  # Remonte à la racine
    model_dir = os.path.join(project_root, "models")  # Dossier unique pour stocker les .pkl

    # Créer le dossier models/ s'il n'existe pas
    os.makedirs(model_dir, exist_ok=True)

    # Sauvegarder le modèle dans le bon dossier
    model_path = os.path.join(model_dir, "linear_regression.pkl")
    joblib.dump(model, model_path)

    print("Successfully trained linear_regression model")