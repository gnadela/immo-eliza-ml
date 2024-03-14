import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer


def get_data():
    df = pd.read_csv("data/properties.csv")
    return df


def impute_missing_values(df, strategy='mean'):
    """
    Impute missing values in a DataFrame using the specified strategy.

    Parameters:
    - df (DataFrame): The DataFrame containing missing values.
    - strategy (str): The imputation strategy. Options are 'mean' (default), 'median', or 'most_frequent'.

    Returns:
    - df_imputed (DataFrame): DataFrame with missing values imputed.
    """
    imputer = SimpleImputer(strategy=strategy)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed

    # Example usage:
    # df_imputed = impute_missing_values(df)


def clean_data(df):
    # Drop columns
    df = df.drop(['latitude', 'longitude', 'id', 'cadastral_income', 'zip_code', 
                              'construction_year', 'primary_energy_consumption_sqm', 'surface_land_sqm'], axis=1)
       
    # Remove outliers 
#    df = df[df['total_area_sqm'] < 5000]
#    df = df[df['price'] < 8000000]

    # Keep max number of frontages to 4
#    df.loc[df['nbr_frontages'] > 4, 'nbr_frontages'] = 4
    df['nbr_frontages'] = df['nbr_frontages'].fillna(1)

    # Keep max number of bedrooms to 12
#   df.loc[df['nbr_bedrooms'] > 12, 'nbr_bedrooms'] = 12
    df['nbr_bedrooms'] = df['nbr_bedrooms'].fillna(0)

    # Fill blank garden and terrace areas with 0
    df['garden_sqm'] = df['garden_sqm'].fillna(0)
    df['terrace_sqm'] = df['terrace_sqm'].fillna(0)

    # Fill blank surface_land_sqm, assume 2 floors
    #df['surface_land_sqm'] = df['garden_sqm'] + df['terrace_sqm'] + df['total_area_sqm'] / 2 

    # Remove rows with blank 
    #df.dropna(subset=['total_area_sqm'], inplace=True)
    #df.dropna(subset=['locality'], inplace=True)

    # Keep only numeric columns
    df = df.select_dtypes(include=['float64', 'int64'])

    return df


def one_hot_encode(df, columns):
    """
    Perform one-hot encoding on specified categorical columns in a DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame containing the columns to be one-hot encoded.
    - columns (list): List of column names to be one-hot encoded.

    Returns:
    - df_encoded (DataFrame): DataFrame with one-hot encoded columns concatenated.
    """
    df_encoded = df.copy()  # Create a copy of the original DataFrame to avoid modifying it directly
    
    for col in columns:
        # Perform one-hot encoding using pandas get_dummies function
        one_hot_encoded = pd.get_dummies(df[col], prefix=col)
        # Concatenate the one-hot encoded columns with the original DataFrame
        df_encoded = pd.concat([df_encoded, one_hot_encoded], axis=1)
        # Drop the original categorical column
        df_encoded.drop(col, axis=1, inplace=True)
    
    return df_encoded

    # Example usage:
    # df_encoded = one_hot_encode(df, ['locality'])


def scale_and_concat(df, columns_to_scale, scaler='min_max'):
    """
    Scale specified numerical columns in a DataFrame and concatenate them with the original DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame containing the columns to be scaled.
    - columns_to_scale (list): List of column names to be scaled.
    - scaler (str): Type of scaler to use, either 'min_max' (default) or 'standard'.

    Returns:
    - df_scaled (DataFrame): DataFrame with scaled columns concatenated.
    """
    #df_scaled = df.copy()  # Create a copy of the original DataFrame to avoid modifying it directly
    
    # Perform scaling
    if scaler == 'min_max':
        scaler = MinMaxScaler()
    elif scaler == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid scaler type. Use 'min_max' or 'standard'.")
    
    scaled_data = scaler.fit_transform(df[columns_to_scale])
    df[columns_to_scale] = scaled_data
    
    return df


def linear_regression(X, y, test_size=0.2, random_state=42):
    """
    Perform linear regression.

    Parameters:
    - X (DataFrame or array-like): Features.
    - y (Series or array-like): Target variable.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int): Random state for reproducibility.

    Returns:
    - model (LinearRegression): Trained linear regression model.
    - X_train, X_test, y_train, y_test: Split datasets.
    - mse (float): Mean squared error.
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scale_and_concat(X_train, ['total_area_sqm', 'garden_sqm', 'terrace_sqm'], scaler='standard')
    scale_and_concat(X_test, ['total_area_sqm', 'garden_sqm', 'terrace_sqm'], scaler='standard')

    # Instantiate and fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

    # Example usage:
    # model, X_train, X_test, y_train, y_test, mse = linear_regression(X, y)

from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a regression model using mean squared error and R-squared score.

    Parameters:
    - model: Trained regression model.
    - X_test (DataFrame or array-like): Test features.
    - y_test (Series or array-like): True test labels.

    Returns:
    - mse (float): Mean squared error.
    - r2 (float): R-squared score.
    """
    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)

    # Calculate R-squared score
    r2 = r2_score(y_test, y_pred)

    return mse, r2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


def plot_learning_curve(estimator, X, y, train_sizes=np.linspace(0.1, 1.0, 50), cv=5):
    """
    Plot learning curves for a given estimator.

    Parameters:
    - estimator: The machine learning model to plot the learning curve for.
    - X: The input features.
    - y: The target labels.
    - train_sizes: The relative or absolute numbers of training examples used to generate the curve.
    - cv: Number of cross-validation folds.

    Returns:
    - None (plots the learning curve).
    """
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, X, y, train_sizes=train_sizes, cv=cv, scoring='neg_mean_squared_error'
    )

    train_scores_mean = -np.mean(train_scores, axis=1)
    validation_scores_mean = -np.mean(validation_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation error')
    plt.xlabel('Number of training examples')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
# plot_learning_curve(model, X_train, y_train)



df = get_data()
#df = one_hot_encode(df, ['property_type', 'locality'])
#print(df.columns)
df = clean_data(df)
df = impute_missing_values(df)
model, X_train, X_test, y_train, y_test = linear_regression(df, df['price'], random_state=42)
mse, r2 = evaluate_model(model, X_test, y_test)
print(mse, r2)
plot_learning_curve(model, X_train, y_train)
plot_learning_curve(model, X_test, y_test)
