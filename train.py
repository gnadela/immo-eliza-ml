import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from xgboost import XGBRegressor
import pickle


# Constants
DATA_FILE = 'data/properties.csv'
TARGET_COLUMN = 'price'
REMOVE_COLUMN = ['id', 'region', 'cadastral_income']


def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess data: handle missing values, encode categorical variables."""
    X = df.drop(columns=[TARGET_COLUMN] + REMOVE_COLUMN)
    y = df[TARGET_COLUMN]

    X = pd.get_dummies(X)               # One-hot encoding
    X.fillna(X.median(), inplace=True)  # Fill missing values

    return X, y

def train_model(X_train, y_train):
    """Train XGBoost model."""
    xgb_model = XGBRegressor(
        n_estimators= 200,        # Increase number of trees
        max_depth=7,             # Increase maximum tree depth
        learning_rate=0.1       # Reduce learning rate
    #    subsample=0.8,           # Use subsample of training instances
    #    colsample_bytree     # Use subsample of features
    #    reg_lambda=1,            # L2 regularization
    #    reg_alpha=0.5            # L1 regularization
    )
    
    xgb_model.fit(X_train, y_train)
    return xgb_model

def save_model(model, file_path):
    """Save the trained model."""
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def main():
    # Load data
    df = load_data(DATA_FILE)

    # Preprocess data
    X, y = preprocess_data(df)

    # Split data into train and test sets
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Save model
    save_model(model, 'trained_model.pkl')


if __name__ == "__main__":
    main()
