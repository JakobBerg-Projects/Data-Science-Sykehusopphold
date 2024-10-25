import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Definer numeriske og kategoriske kolonner
numeric_cols = [
    'alder', 'utdanning', 'blodtrykk', 'hvite_blodlegemer', 'hjertefrekvens',
    'respirasjonsfrekvens', 'kroppstemperatur', 'lungefunksjon', 'serumalbumin',
    'kreatinin', 'natrium', 'blod_ph', 'antall_komorbiditeter', 'koma_score',
    'fysiologisk_score', 'apache_fysiologisk_score', 'overlevelsesestimat_2mnd',
    'overlevelsesestimat_6mnd', 'lege_overlevelsesestimat_2mnd',
    'lege_overlevelsesestimat_6mnd', 'diabetes', 'glukose', 'blodurea_nitrogen',
    'urinmengde', 'demens'
]
categorical_cols = ['kjønn', 'etnisitet', 'sykdomskategori', 'sykdom_underkategori', 'dnr_status', 'inntekt', 'kreft']

def create_severity_indicators(df):
    df['alvorlighetsgrad_høy'] = (df['fysiologisk_score'] > 60).astype(bool)
    df['alvorlighetsgrad_lav'] = (df['fysiologisk_score'] < 10).astype(bool)
    df['alvorlighetsgrad_middels'] = ((df['fysiologisk_score'] >= 10) & (df['fysiologisk_score'] <= 60)).astype(bool)
    return df

def prepare_data_for_length_prediction(df, prediction_mode=False):
    """
    Forbereder data for prediksjon av sykehusoppholdslengde.
    """
    df = create_severity_indicators(df)
    df['alder_fysiologisk_interaction'] = df['alder'] * df['fysiologisk_score']
    df['age_binned'] = pd.cut(df['alder'], bins=[0, 30, 60, np.inf], labels=['young', 'middle-aged', 'senior'])
    
    # Ekskluder målvariabelen 'oppholdslengde' i treningsmodus
    y = df['oppholdslengde'] if not prediction_mode else None
    if not prediction_mode:
        df = df.drop(columns=['oppholdslengde'])

    # Tilleggs kolonner
    additional_numeric_cols = ['alder_fysiologisk_interaction']
    additional_categorical_cols = ['age_binned']
    numeric_cols_all = numeric_cols + additional_numeric_cols
    categorical_cols_all = categorical_cols + additional_categorical_cols

    return df, numeric_cols_all, categorical_cols_all, y

def prepare_data_for_death_classification(df, prediction_mode=False):
    """
    Forbereder data for klassifikasjon av sykehusdød.
    """
    df = create_severity_indicators(df)
    df['alder_fysiologisk_interaction'] = df['alder'] * df['fysiologisk_score']
    df['age_binned'] = pd.cut(df['alder'], bins=[0, 30, 60, np.inf], labels=['young', 'middle-aged', 'senior'])
    
    # Ekskluder 'sykehusdød' som målvariabel i treningsmodus
    y = df['sykehusdød'] if not prediction_mode else None
    if not prediction_mode:
        df = df.drop(columns=['sykehusdød'])

    # Tilleggs kolonner
    additional_numeric_cols = ['alder_fysiologisk_interaction']
    additional_categorical_cols = ['age_binned']
    numeric_cols_all = numeric_cols + additional_numeric_cols
    categorical_cols_all = categorical_cols + additional_categorical_cols

    return df, numeric_cols_all, categorical_cols_all, y



def get_col_transformer(numeric_cols, categorical_cols):
    num_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer()),  
        ('scaler', StandardScaler())  
    ])

    cat_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),  
        ('one-hot-encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  
    ])
    
    return ColumnTransformer(transformers=[
        ('num_pipeline', num_pipeline, numeric_cols),
        ('cat_pipeline', cat_pipeline, categorical_cols)
    ])