import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


def split_columns(X: pd.DataFrame):
    """
    Splits the DataFrame columns into numerical, multi-class categorical, and binary categorical.
    Parameters:
    X (pd.DataFrame): The input DataFrame.
    Returns:
    Three lists - numerical columns, multi-class categorical, binary categorical.
    """
    cat_cols = X.select_dtypes(include=['object']).columns
        
    binary_cat_cols = []    
    multi_cat_cols = []     
    num_cols = []           

    for col in cat_cols:
        if X[col].nunique() == 2:
            binary_cat_cols.append(col)
        else:
            multi_cat_cols.append(col)

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    return num_cols, multi_cat_cols, binary_cat_cols


def data_preprocessing():
    """
    Preprocesses the data by importing, feature engineering, splitting columns, and creating a pipeline.
    Returns:
    X (pd.DataFrame): Features after preprocessing.
    y (pd.Series): Target variable.
    pipeline (Pipeline): Preprocessing pipeline.
    """
    
    #data import
    data = pd.read_csv("data.csv")
    col_to_drop = ["RowNumber", "CustomerId", "Surname", "Complain", "Exited"]
    X = data.drop(col_to_drop, axis=1)
    y = data['Exited']

    #new features
    X['BalanceSalaryRatio'] = np.where( X['EstimatedSalary'] == 0, 0, X['Balance'] / X['EstimatedSalary'])
    X['TenureByAge'] = X['Tenure'] / X['Age']
    X['CreditScoreGivenAge'] = X['CreditScore'] / X['Age']

    num_cols, multi_cat_cols, binary_cat_cols = split_columns(X)

    # column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_cols),
            ('multi', OneHotEncoder(handle_unknown='ignore', sparse_output=False), multi_cat_cols),
            ('binary', OrdinalEncoder(), binary_cat_cols)
        ],
    
        verbose_feature_names_out=False # to avoid prefixes in feature names
    )

    # pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),  
        ('scaler', StandardScaler())      
    ])

    return X, y, pipeline