from sklearn.linear_model import LinearRegression
import joblib

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, "models/linear_regression.pkl")
    print("Successfully trained linear_regression model")