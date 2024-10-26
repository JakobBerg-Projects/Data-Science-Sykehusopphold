# Definer numeriske og kategoriske kolonner
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

numeric_cols = [
    'alder', 'utdanning', 'blodtrykk', 'hvite_blodlegemer', 'hjertefrekvens',
    'respirasjonsfrekvens', 'kroppstemperatur', 'lungefunksjon', 'serumalbumin',
    'kreatinin', 'natrium', 'blod_ph', 'antall_komorbiditeter', 'koma_score',
    'fysiologisk_score', 'apache_fysiologisk_score', 'overlevelsesestimat_2mnd',
    'overlevelsesestimat_6mnd', 'glukose', 'blodurea_nitrogen',
    'urinmengde'
]
categorical_cols = ['kjønn', 'etnisitet', 'sykdomskategori', 'sykdom_underkategori', 'dnr_status', 'inntekt', 'demens', 'diabetes']

def create_severity_indicators(df):
    # Define severity levels based on 'fysiologisk_score'
    df['alvorlighetsgrad'] = pd.cut(
        df['fysiologisk_score'],
        bins=[-np.inf, 10, 60, np.inf],
        labels=['lav', 'middels', 'høy']
    )
    return df
def create_omfattende_behandling(df):
    df['omfattende_behandling'] = (
        (df['koma_score'] == 0) & 
        (df['antall_komorbiditeter'] >= 3) & 
        (df['fysiologisk_score'].between(10, 60))
    )
    return df

def prepare_data_for_length_prediction(df, sykehusdod_model=None, prediction_mode=False):
    """
    Prepares data for predicting hospital stay length. If 'sykehusdød' is missing, it is imputed using the provided classification model.
    """
    df = create_severity_indicators(df)
    df = create_omfattende_behandling(df)
    df['alder_fysiologisk_interaction'] = df['alder'] * df['fysiologisk_score']
    df['age_binned'] = pd.cut(df['alder'], bins=[0, 30, 60, np.inf], labels=['young', 'middle-aged', 'senior'])

    # If prediction_mode is True and 'sykehusdød' is missing, impute it using the classification model
    if prediction_mode and 'sykehusdød' not in df.columns and sykehusdod_model:
        X_classification, _, _, _ = prepare_data_for_death_classification(df, prediction_mode=True)
        df['sykehusdød'] = sykehusdod_model.predict(X_classification)
    
    # Target variable 'oppholdslengde' for length prediction
    y = df['oppholdslengde'] if not prediction_mode else None
    if not prediction_mode:
        df = df.drop(columns=['oppholdslengde'])  # Don't drop 'sykehusdød', keep it as a feature

    # Additional columns
    additional_numeric_cols = ['alder_fysiologisk_interaction']
    additional_categorical_cols = ['age_binned', 'sykehusdød', 'alvorlighetsgrad', 'omfattende_behandling']  # Ensure 'sykehusdød' is in categorical features
    numeric_cols_all = numeric_cols + additional_numeric_cols
    categorical_cols_all = categorical_cols + additional_categorical_cols

    return df, numeric_cols_all, categorical_cols_all, y


def prepare_data_for_death_classification(df, prediction_mode=False):
    """
    Forbereder data for klassifikasjon av sykehusdød.
    """
    df = create_severity_indicators(df)
    df = create_omfattende_behandling(df)
    df['alder_fysiologisk_interaction'] = df['alder'] * df['fysiologisk_score']
    df['age_binned'] = pd.cut(df['alder'], bins=[0, 30, 60, np.inf], labels=['young', 'middle-aged', 'senior'])
    
    # Ekskluder 'sykehusdød' som målvariabel i treningsmodus
    y = df['sykehusdød'] if not prediction_mode else None
    if not prediction_mode:
        df = df.drop(columns=['sykehusdød'])

    # Tilleggs kolonner
    additional_numeric_cols = ['alder_fysiologisk_interaction']
    additional_categorical_cols = ['age_binned', 'alvorlighetsgrad', 'omfattende_behandling']
    numeric_cols_all = numeric_cols + additional_numeric_cols
    categorical_cols_all = categorical_cols + additional_categorical_cols

    return df, numeric_cols_all, categorical_cols_all, y



def get_col_transformer(numeric_cols, categorical_cols, passthrough_cols):
    num_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer()),
        ('scaler', StandardScaler())  
    ])

    cat_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),  
        ('one-hot-encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  
    ])

    # Del de kategoriske kolonnene mellom de som skal one-hot encodes og de som skal passeres gjennom
    categorical_cols_to_encode = [col for col in categorical_cols if col not in passthrough_cols]

    return ColumnTransformer(transformers=[
        ('num_pipeline', num_pipeline, numeric_cols),
        ('cat_pipeline', cat_pipeline, categorical_cols_to_encode),
        ('passthrough', 'passthrough', passthrough_cols)  # Behandle spesifikke kolonner som 'passthrough'
    ])