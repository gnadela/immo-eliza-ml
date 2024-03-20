import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PowerTransformer

TARGET_COLUMN = 'price'
REMOVE_COLUMN = ['id', 'region', 'cadastral_income']

def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess data: handle missing values, encode categorical variables."""
    X = df.drop(columns=[TARGET_COLUMN] + REMOVE_COLUMN)
    y = df[TARGET_COLUMN]
    
    X = pd.get_dummies(X)
    X.fillna(X.median(), inplace=True)  # Fill missing values

    return X, y

def load_model(file_path):
    """Load the trained model."""
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(model, X_test):
    """Make predictions using the trained model."""
    return model.predict(X_test)

def evaluate(y_test, y_pred):
    """Evaluate the predictions."""
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("Evaluation Metrics:")
    print("Root Mean Squared Error:", rmse)
    print("R-squared:", r2)

def perform_cross_validation(model, X_train, y_train):
    """Perform k-fold cross-validation."""
    cv_scores = cross_val_score(model, X_train, y_train, cv=6, scoring='neg_mean_squared_error')
    rmse_cv_scores = np.sqrt(-cv_scores)
    mean_rmse_cv = rmse_cv_scores.mean()
    print('')
    print('Cross-Validation Results:')
    print("Mean Cross-Validation RMSE:", mean_rmse_cv)
    print("Cross-Validation RMSE Scores:", rmse_cv_scores)  

def main():
    # Load data
    df = load_data('data/properties.csv')

    # Preprocess data
    X, y = preprocess_data(df)

    # Load trained model
    model = load_model('model/trained_model.pkl')

    # Make predictions
    predictions = predict(model, X)

    # Evaluate model
    evaluate(y, predictions)

    # Perform cross-validation
    perform_cross_validation(model, X, y)

if __name__ == "__main__":
    main()
