from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def test_model(pipeline, X_test, y_test):
    # Make predictions on the testing data
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R-squared:", r2)
