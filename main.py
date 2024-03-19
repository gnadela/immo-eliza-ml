import pandas as pd
from sklearn.model_selection import train_test_split
from train import load_data, preprocess_data, train_model, perform_cross_validation
from predict import evaluate_model, plot_results

# Constants
DATA_FILE = 'data/properties.csv'

def main():
    # Load data
    df = load_data(DATA_FILE)

    # Preprocess data
    X, y = preprocess_data(df)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    y_pred = evaluate_model(model, X_test, y_test)

    # Perform cross-validation
    perform_cross_validation(model, X_train, y_train)

    # Plot results
    # plot_results(y_test, y_pred)


if __name__ == "__main__":
    main()

