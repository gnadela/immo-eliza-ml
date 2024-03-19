import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Print evaluation metrics
    print("Evaluation Metrics:")
    print("Root Mean Squared Error:", rmse)
    print("R-squared:", r2)

    return y_pred

def plot_results(y_test, y_pred):
    # Calculate residuals
    residuals = y_test - y_pred

    # Plot actual vs. predicted values
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
    plt.title('Actual vs. Predicted')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.grid(True)
    plt.show()

    # Plot residuals
    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred, residuals, color='green', alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Residuals')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residual')
    plt.grid(True)
    plt.show()

