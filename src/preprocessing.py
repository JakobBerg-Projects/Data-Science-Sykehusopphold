# Definer numeriske og kategoriske kolonner
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
    'overlevelsesestimat_6mnd', 'glukose', 'blodurea_nitrogen',
    'urinmengde'
]
categorical_cols = ['kjønn', 'etnisitet', 'sykdomskategori', 'sykdom_underkategori', 'dnr_status', 'inntekt', 'demens', 'diabetes']

def create_severity_indicators(df):
   # Definer alvorlighetsgrader basert på 'fysiologisk_score'
    df['alvorlighetsgrad'] = pd.cut(
        df['fysiologisk_score'],
        bins=[-np.inf, 10, 60, np.inf],
        labels=['lav', 'middels', 'høy']
    )
    return df
def create_omfattende_behandling(df):
    # Opprett indikator for omfattende behandling basert på koma, komorbiditet og fysiologisk score
    df['omfattende_behandling'] = (
        (df['koma_score'] == 0) & 
        (df['antall_komorbiditeter'] >= 3) & 
        (df['fysiologisk_score'].between(10, 60))
    )
    return df

def gjennomsnitt__oppholdslengde_sykdom(df, avg_length_dict):
    # Bruk den forhåndsberegnede dictionary til å mappe gjennomsnittsverdiene
    df['gjennomsnitt_oppholdslengde_sykdom_underkategori'] = df['sykdom_underkategori'].map(avg_length_dict)
    return df

def ingen_dnr(df):
    # Opprett indikator for rader hvor 'dnr_status' opprinnelig var manglende
    df['ingen dnr'] = df['dnr_status'].isna()
    
    # Erstatt manglende verdier i 'dnr_status' med 'ingen dnr'
    df['dnr_status'] = df['dnr_status'].fillna('ingen dnr')
    
    return df

def prepare_data_for_length_prediction(df, avg_length_dict, sykehusdod_model=None, prediction_mode=False):
    df = create_severity_indicators(df)
    df = create_omfattende_behandling(df)
    df = gjennomsnitt__oppholdslengde_sykdom(df, avg_length_dict)
    df = ingen_dnr(df)
    df['alder_fysiologisk_interaction'] = df['alder'] * df['fysiologisk_score']
    df['age_binned'] = pd.cut(df['alder'], bins=[0, 30, 60, np.inf], labels=['ung', 'middelaldrende', 'eldre'])

    # Imputer 'sykehusdød' hvis den mangler og 'prediction_mode' er aktivert
    if prediction_mode and 'sykehusdød' not in df.columns and sykehusdod_model:
        X_classification, _, _, _ = prepare_data_for_death_classification(df, prediction_mode=True)
        df['sykehusdød'] = sykehusdod_model.predict(X_classification)
    
    # Sett 'oppholdslengde' som målvariabel hvis ikke prediksjonsmodus
    y = df['oppholdslengde'] if not prediction_mode else None
    if not prediction_mode:
        df = df.drop(columns=['oppholdslengde'])  # Behold 'sykehusdød' som funksjon

    # Ekstra kolonner for numerisk og kategorisk analyse
    additional_numeric_cols = ['alder_fysiologisk_interaction', 'gjennomsnitt_oppholdslengde_sykdom_underkategori']
    additional_categorical_cols = ['age_binned', 'sykehusdød', 'alvorlighetsgrad', 'omfattende_behandling']  # Ensure 'sykehusdød' is in categorical features
    numeric_cols_all = numeric_cols + additional_numeric_cols
    categorical_cols_all = categorical_cols + additional_categorical_cols

    return df, numeric_cols_all, categorical_cols_all, y


def prepare_data_for_death_classification(df, prediction_mode=False):
    df = create_severity_indicators(df)
    df = create_omfattende_behandling(df)
    df['alder_fysiologisk_interaction'] = df['alder'] * df['fysiologisk_score']
    df['age_binned'] = pd.cut(df['alder'], bins=[0, 30, 60, np.inf], labels=['young', 'middle-aged', 'senior'])
    
    # Sett 'sykehusdød' som målvariabel med mindre det er prediksjonsmodus
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
    # Numerisk pipeline med mean-imputering og StandardScaler
    num_pipeline_mean = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='mean')), 
        ('scaler', StandardScaler())  
    ])

    # Numerisk pipeline med median-imputering for spesifikke variabler
    num_pipeline_median = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())  
    ])
    
    # Numerisk pipeline med modus-imputering for spesifikke diskrete variabler
    num_pipeline_mode = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler())  
    ])
    
    # Kategorisk pipeline med modus-imputering og OneHotEncoding
    cat_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('one-hot-encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  
    ])
    
    # Pipeline for passthrough kolonner som kun skal imputeres
    passthrough_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent'))
    ])

    # Kolonner for spesifikke imputeringsteknikker
    median_impute_cols = ['alder', 'lege_overlevelsesestimat_2mnd', 'lege_overlevelsesestimat_6mnd']
    mode_impute_cols = ['adl_stedfortreder']

    # Kategoriske kolonner som skal one-hot-encodes
    categorical_cols_to_encode = [col for col in categorical_cols if col not in passthrough_cols]

    col_transformer = ColumnTransformer(transformers=[
        ('num_pipeline_mean', num_pipeline_mean, [col for col in numeric_cols if col not in median_impute_cols and col not in mode_impute_cols]),
        ('num_pipeline_median', num_pipeline_median, median_impute_cols),
        ('num_pipeline_mode', num_pipeline_mode, mode_impute_cols),
        ('cat_pipeline', cat_pipeline, categorical_cols_to_encode),
        ('passthrough_impute', passthrough_pipeline, passthrough_cols)  # Imputer passthrough kolonner uten encoding
    ])

    return col_transformer