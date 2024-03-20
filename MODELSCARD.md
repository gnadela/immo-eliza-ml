# Model card

## Project context

This project aims to create a machine learning model to predict real estate property prices in Belgium for a fictive real estate company Immo Eliza. The model development involves data preprocessing, model selection, training, evaluation, cross-validation and iteration.

## Data

The input dataset consists of real estate property information scraped from Immoweb, the largest real estate website in Belgium. 

Some preliminary prepation of the input data:
- There are about 76 000 properties, roughly equally spread across houses and apartments
- Each property has a unique identifier id

- The target variable is price
- Variables prefixed with fl_ are dummy variables (1/0)
- Variables suffixed with _sqm indicate the measurement is in square meters
- All missing categories for the categorical variables are encoded as MISSING

## Model details

The model includes additional pre-processing steps on the input data:
- Removal of insignificant columns (id, region, cadastral_income)
- New feature addition: postal_zone (taking the first 2 digits of zip_code)
- Handling of NaNs by replacing them with median values
- Using one-hot encoding to convert categorical data into numeric features
- Rescaling numeric features using PowerTransformer

The final model uses **XGBoost regressor**. This after testing Linear Regression and Random Forest.  

The model splits the data set into training and testing set (80/20).

XGBoost hyperparameters for number of trees (n_estimators), tree depth (max_depth), and learning rate(learning_rate) were tuned.

The trained model is stored using Pickle.

## Performance

Performance metrics for the **XGBoost** model:

    Root Mean Squared Error (RMSE): 141932.9963525912
    R-squared (R2): 0.8951632610887943

K-fold cross-validation is performed to assess ths model's generalization performance:

    Mean Cross-Validation RMSE: 230398.6343194463

    Cross-Validation RMSE Scores: 
    [186083.57247229 200757.64226223 205611.01100654 230495.30474894 282815.97321565 276628.30221102]

**Linear Regression** was used as the baseline for the model before adopting XGBoost.

Performance metrics for the Linear Regression model:

    Root Mean Squared Error: 292475.3457237516
    R-squared: 0.506795723844395

**Random Forest** was also tested. It showed similar results as XGBoost but was taking a lot more time to run, ie. about 40 minutes compared to 5 seconds with XGBoost.




## Limitations

- The model's performance heavily relies on the quality and representativeness of the input data.
- The model may overfit or underfit the training data depending on hyperparameters tuning.

## Usage

Dependencies:

    pandas
    numpy
    scikit-learn
    xgboost

Scripts:

- *train.py*: Loads and preprocesses the data, trains the XGBoost model, evaluates the model, performs cross-validation, and plots results.

- *predict.py*: Uses the trained model to predict the price of a new house.
