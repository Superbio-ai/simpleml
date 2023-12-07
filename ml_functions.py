from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate, train_test_split

from pandas import DataFrame
import numpy as np
import pandas as pd

#####################

def check_inputs(problem_type: str, model_type: str, predictors: list, target: str, train_test_ratio: float, missing_treatment: str, df: DataFrame):
    col_names = list(df.columns)
    if problem_type not in ['regression','classification']:
        raise ValueError("Must specify either regression or classification problem_type")
    if model_type not in ['linear_model','random_forest','gradient_boosting','neural_network','best']:
        raise ValueError("Must specify either linear_model random_forest, gradient_boosting, neural_network, or best model_type")
    if not set(predictors).issubset(col_names):
        raise ValueError("Not all specified predictors are found in the column names")
    if target not in col_names:
        raise ValueError("Target not found in column names")
    if target in predictors:
        raise ValueError("Target should not be included in predictors")
    if len(predictors) == 0:
        raise ValueError("At least one predictor should be provided")
    if train_test_ratio>0.9 or train_test_ratio<0.1:
        raise ValueError("Please specify a train_test_ratio between 0.1 and 0.9")
    if missing_treatment not in ['mean','drop']:
        raise ValueError("Please specify whether to drop rows with missing values, or replace missing values with column mean")


def _get_model(problem_type: str, model_type: str):
    if problem_type == 'regression':
        if model_type == 'linear_model':
            model = LinearRegression()
        elif model_type == 'random_forest':
            model = RandomForestRegressor()
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor()
        elif model_type == 'neural_network':
            model = MLPRegressor()
    if problem_type == 'classification':
        if model_type == 'linear_model':
            model = LogisticRegression()
        elif model_type == 'random_forest':
            model = RandomForestClassifier()
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier()
        elif model_type == 'neural_network':
            model = MLPClassifier()
    return model


def run_simple_pipelines(df: DataFrame, problem_type: str, model_type: str, predictors: list, target: str, train_test_ratio: float = 0.8, folds: int = 5, missing_treatment: str = 'mean'):    
    
    #filter dataframe to only predictors (need to sort out target later)
    input_df = df[predictors]
    
    #check which are numeric vs categorical
    numeric_features = input_df.select_dtypes('number').columns
    categorical_features = input_df.select_dtypes('object').columns
    
    #impute missing data
    if missing_treatment == 'mean':
        imp_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')
        imp_mean.fit(input_df)
        input_df = imp_mean.transform(input_df)
        target = df[target]
    elif missing_treatment == 'drop':
        input_df.dropna(inplace=True)
        indices = input_df.index
        target = df[target][indices]
    
    #set up pipelines
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())            
            ]
        )
    
    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(drop='if_binary', max_categories=5, sparse=False, dtype='int', handle_unknown="ignore"))            
            ]
        )
    
    col_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    # main pipeline, including modelling
    if model_type != 'best':
        model = _get_model(problem_type, model_type)
        #add hyperparameter search another time, only if needed
        main_pipe = Pipeline(
            steps=[
                ("preprocessor", col_transformer),
                ("model", model)
            ]
        )
    #add best another time, and only if needed
    
    #splitting data
    X_train, X_test, y_train, y_test = train_test_split(input_df, target, test_size=(1-train_test_ratio), random_state=42)
    
    #getting cross-validation scores
    with_scores = cross_validate(main_pipe, X_train, y_train, return_train_score = True, cv = folds)
    scores = pd.DataFrame(with_scores)
    
    #add fit on whole data only if needed
    
    #simple info for response
    means = scores.mean()
    
    return means









'''

clf = make_pipeline(StandardScaler(), 
                    GridSearchCV(LogisticRegression(),
                                 param_grid={'logisticregression__C': [0.1, 10.]},
                                 cv=2,
                                 refit=True))

clf.fit()
clf.predict()
'''