from src.train import main as train_main
from src.predict import main as predict_main

def main():
    # Train the model
    train_main()

    # Make predictions
    predict_main()

if __name__ == "__main__":
    main()
