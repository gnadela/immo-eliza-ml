from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

def preprocess_data(X):

    # Features to drop
    X = X.drop(columns=['cadastral_income', 'region']) 

    # Convert to str to be included in categorical feature
    X['zip_code'] = X['zip_code'].astype(str)

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
