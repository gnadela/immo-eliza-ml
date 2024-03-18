from train import train_model
from test import test_model
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('data/properties.csv')

# Define features and target variable
X = df.drop(columns=['id', 'cadastral_income', 'region']) 
y = df['price']  # Target variable

# New Feature: Extract the first two digits of the zip_code 
X['zip_code'] = X['zip_code'].astype(str)
X['postal_zone'] = X['zip_code'].str[:2]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=155)

# Train the model
pipeline = train_model(X_train, y_train)

# Test the model
test_model(pipeline, X_test, y_test)
