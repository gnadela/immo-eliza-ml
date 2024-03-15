from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from preprocessing import preprocess_data

def train_model(X_train, y_train):
    # Create preprocessing pipeline
    preprocessor = preprocess_data(X_train)

    # Define the model
    model = LinearRegression()

    # Create the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Fit the pipeline (preprocessing + model) on the training data
    pipeline.fit(X_train, y_train)

    return pipeline
