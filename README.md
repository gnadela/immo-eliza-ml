# Immo Eliza Machine Learning Project
## Introduction

This project aims to develop a machine learning model for predicting real estate property prices in Belgium for the real estate company Immo Eliza. The model utilizes the XGBoost algorithm for regression tasks and involves various data preprocessing steps, model training, evaluation, and iteration to achieve accurate predictions.


## Repo Structure
```
.
├── data/
│ ├── properties.csv
├── analysis/
│ ├── analysis.ipynb
├── src/
│ ├── train.py
│ ├── predict.py
├── main.py
├── model/
│ ├── trained_model.pkl
├── .gitignore
├── requiremets.txt
├── MODELSCARD.md
└── README.md
```


## Data

The input dataset (properties.csv) contains information about real estate properties in Belgium. This data is scraped from Immoweb, the largest real estate website in Belgium. 

There are about 76 000 properties, roughly equally spread across houses and apartments. 
It includes 30 features:
```
['id', 'price', 'property_type', 'subproperty_type', 'region', 'province', 'locality', 'zip_code', 'latitude', 'longitude', 'construction_year', 'total_area_sqm', 'surface_land_sqm', 'nbr_frontages', 'nbr_bedrooms', 'equipped_kitchen', 'fl_furnished', 'fl_open_fire', 'fl_terrace', 'terrace_sqm', 'fl_garden', 'garden_sqm', 'fl_swimming_pool', 'fl_floodzone', 'state_building', 'primary_energy_consumption_sqm', 'epc', 'heating_type', 'fl_double_glazing', 'cadastral_income']
```
The target variable is the **price** of the property.

Some preliminary preparation of the data:
- Variables prefixed with fl_ are dummy variables (1/0)
- Variables suffixed with _sqm indicate the measurement is in square meters
- All missing categories for the categorical variables are encoded as MISSING


## Model Development
### Data Preprocessing

The model includes additional pre-processing steps on the input data:
- Removal of insignificant columns (id, region, cadastral_income)
- New feature addition: postal_zone (taking the first 2 digits of zip_code)
- Handling of NaNs by replacing them with median values
- Using one-hot encoding to convert categorical data into numeric features
- Rescaling numeric features using PowerTransformer

### Model Selection and Training

Linear Regression, Random Forest, and XGBoost were evaluated (see MODELSCARD for details).

The final model employs the **XGBoost Regression** algorithm, chosen for its efficiency and performance. The model is trained using the preprocessed data.


### Model Evaluation

- Evaluation Metrics: Performance of the trained model is assessed using metrics such as 
    - Root Mean Squared Error (RMSE)
    - R-squared (R2)
    - Mean Absolute Error (MAE)

- Cross-validation: K-fold cross-validation is performed to validate the model's performance and ensure its generalization ability.

### Model Iteration

- Hyperparameter Tuning: Various hyperparameters of the XGBoost model are tuned to optimize performance:

    - *n_estimators:* the number of boosting rounds or trees to build
 
    - *max_depth:* the maximum depth of each tree, deeper trees may capture more complex patterns 
 
    - *learning_rate:* the step size shrinkage used to prevent overfitting. 



## Usage

Clone the repository:

```
git clone https://github.com/gnadela/immo-eliza-ml
cd immo-eliza-ml
```

Install dependencies:
```
pip install -r requirements.txt
```
Run the main script

```
python main.py
```
Alternatively, the scripts can be run separately:

- Training the Model:
        Run *train.py* to train the model using the provided dataset (properties.csv).
        The trained model will be saved as *trained_model.pkl* in the model/ directory.

```
    python train.py
```
- Making Predictions:
        Run *predict.py* to load the trained model (*trained_model.pkl*) and make predictions on new data.
```
    python predict.py
```
## Timeline
This project took 1 week for completion.

## Personal Situation

This project was done as part of the AI Bootcamp at BeCode.org. This is my first machine learning modeling project.

Connect with me on [LinkedIn](https://www.linkedin.com/in/geraldine-nadela-60827a11/).

