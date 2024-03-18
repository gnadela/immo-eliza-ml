from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

def preprocess_data(X):

    # Define categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist() 
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Define preprocessing steps for categorical and numerical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', PowerTransformer())
    ])

    # Combine preprocessing steps for categorical and numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ])

    return preprocessor


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
