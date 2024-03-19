# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

# Constants
TARGET_COLUMN = 'price'

def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess data: handle missing values, encode categorical variables."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    X = pd.get_dummies(X)
    X.fillna(0, inplace=True)  # Fill missing values
    return X, y

def train_model(X_train, y_train):
    """Train XGBoost model."""
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)
    return xgb_model

def perform_cross_validation(model, X_train, y_train):
    """Perform k-fold cross-validation."""
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_cv_scores = np.sqrt(-cv_scores)
    mean_rmse_cv = rmse_cv_scores.mean()
    print('')
    print('Cross-Validation Results:')
    print("Mean Cross-Validation RMSE:", mean_rmse_cv)
    print("Cross-Validation RMSE Scores:", rmse_cv_scores)
