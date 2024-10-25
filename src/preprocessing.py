import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer

numeric_cols = [
        'alder', 'utdanning', 'blodtrykk', 'hvite_blodlegemer', 'hjertefrekvens', 
        'respirasjonsfrekvens', 'kroppstemperatur', 'lungefunksjon', 'serumalbumin', 
        'kreatinin', 'natrium', 'blod_ph', 
        'antall_komorbiditeter', 'koma_score', 'fysiologisk_score', 
        'apache_fysiologisk_score', 'overlevelsesestimat_2mnd' ,'overlevelsesestimat_6mnd', 
        'lege_overlevelsesestimat_2mnd', 'lege_overlevelsesestimat_6mnd', 'diabetes', 
        'glukose', 'blodurea_nitrogen', 'urinmengde',
        'demens'
    ]
categorical_cols = ['kjønn', 'etnisitet', 'sykdomskategori', 
                        'sykdom_underkategori', 'dnr_status',
                     'inntekt', 'kreft']



def create_severity_indicators(df):
    df['alvorlighetsgrad_høy'] = (df['fysiologisk_score'] > 60).astype(bool)
    df['alvorlighetsgrad_lav'] = (df['fysiologisk_score'] < 10).astype(bool)
    df['alvorlighetsgrad_middels'] = ((df['fysiologisk_score'] >= 10) & (df['fysiologisk_score'] <= 60)).astype(bool)
    return df

def prepare_data_for_prediction(df, prediction_mode=False):
    # Step 1: Process ADL (if relevant for the new input)

    # Step 2: Create severity level indicators
    df = create_severity_indicators(df)
    
    # Step 3: Additional features (e.g., interaction terms, ratios)
    df['alder_fysiologisk_interaction'] = df['alder'] * df['fysiologisk_score']
    
    # Example of binning age
    df['age_binned'] = pd.cut(df['alder'], bins=[0, 30, 60, np.inf], labels=['young', 'middle-aged', 'senior'])
    
    if not prediction_mode:
        # Step 4: Extract the target column (e.g., 'oppholdslengde') only during training mode
        y = df['oppholdslengde']  # Assuming 'oppholdslengde' is the target variable
        df = df.drop(columns=['oppholdslengde'])  # Drop the target from the DataFrame
    else:
        y = None  # No target column during prediction

    # Additional columns to treat as categorical or numeric
    additional_numeric_cols = ['alder_fysiologisk_interaction', 'bilirubin_kreatinin_ratio']
    additional_categorical_cols = ['age_binned']

    # Combine all numeric and categorical features
    numeric_cols_all = numeric_cols + additional_numeric_cols
    categorical_cols_all = categorical_cols + additional_categorical_cols

    # Return the modified DataFrame, numeric, and categorical columns
    return df, numeric_cols_all, categorical_cols_all, y

# Pipeline for numeric and categorical preprocessing
num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer()),  # Placeholder for numeric imputation
    ('scaler', StandardScaler())  # Scaling for numeric features
])

cat_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),  # Categorical imputation
    ('one-hot-encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-hot encoding
])

# ColumnTransformer for applying preprocessing pipelines
col_transformer = ColumnTransformer(transformers=[
    ('num_pipeline', num_pipeline, numeric_cols),  # Numeric pipeline for numeric columns
    ('cat_pipeline', cat_pipeline, categorical_cols)  # Categorical pipeline for categorical columns
])
