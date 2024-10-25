import numpy as np
import pandas as pd  # Import pandas for creating DataFrame
import pickle
from flask import Flask, request, render_template
from waitress import serve
from preprocessing import prepare_data_for_prediction, col_transformer

app = Flask(__name__)

# Load the trained model from 'model.pkl'
model, col_transformer, feature_names= pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ''' 
    Render the prediction result on HTML
    '''
    # Get data from form
    features = dict(request.form)
    
    try:
        # Extract all input values from the form
        input_data = {}
        
        # Categorical data
        sub_disease = features.get('sykdom_underkategori', 'other')
        input_data.update({
            'kjønn': [features.get('kjønn', 'male')],  # Default to 'male'
            'etnisitet': [features.get('etnisitet', 'white')],  # Default to 'white'
            'sykdom_underkategori': [sub_disease],  # Use the selected sub-disease
            'dnr_status': [features.get('dnr_status', 'no')],  # Default to 'no'
            'inntekt': [features.get('inntekt', 'unknown')],  # Default to 'unknown'
            'kreft': [features.get('kreft', 'no')]  # Default to 'no'
        })

        # Automatically assign Disease Category based on the Sub-disease Category
        sub_disease_to_disease_mapping = {
            'sykdom_ARF/MOSF w/Sepsis': 'ARF/MOSF',
            'sykdom_hjertesvikt': 'COPD/CHF/Cirrhosis',
            'sykdom_kols': 'COPD/CHF/Cirrhosis',
            'sykdom_levercirrhose': 'COPD/CHF/Cirrhosis',
            'sykdom_tykktarmskreft': 'Cancer',
            'sykdom_koma': 'Coma',
            'sykdom_lungekreft': 'Cancer',
            'sykdom_flerorgansvikt_malignt': 'ARF/MOSF'
        }
        sykdomskategori = sub_disease_to_disease_mapping.get(sub_disease, 'other')
        input_data.update({
            'sykdomskategori': [sykdomskategori]
        })

        # Numeric data
        input_data.update({
            'alder': [float(features.get('alder', 0))],
            'utdanning': [float(features.get('utdanning', 0))],
            'blodtrykk': [float(features.get('blodtrykk', 0))],
            'hvite_blodlegemer': [float(features.get('hvite_blodlegemer', 0))],
            'hjertefrekvens': [float(features.get('hjertefrekvens', 0))],
            'respirasjonsfrekvens': [float(features.get('respirasjonsfrekvens', 0))],
            'kroppstemperatur': [float(features.get('kroppstemperatur', 0))],
            'lungefunksjon': [float(features.get('lungefunksjon', 0))],
            'serumalbumin': [float(features.get('serumalbumin', 0))],
            'kreatinin': [float(features.get('kreatinin', 0))],
            'natrium': [float(features.get('natrium', 0))],
            'blod_ph': [float(features.get('blod_ph', 0))],
            'glukose': [float(features.get('glukose', 0))],
            'blodurea_nitrogen': [float(features.get('blodurea_nitrogen', 0))],
            'urinmengde': [float(features.get('urinmengde', 0))],
            'antall_komorbiditeter': [float(features.get('antall_komorbiditeter', 0))],
            'koma_score': [float(features.get('koma_score', 0))],
            'fysiologisk_score': [float(features.get('fysiologisk_score', 0))],
            'apache_fysiologisk_score': [float(features.get('apache_fysiologisk_score', 0))],
            'overlevelsesestimat_2mnd': [float(features.get('overlevelsesestimat_2mnd', 0))],
            'overlevelsesestimat_6mnd': [float(features.get('overlevelsesestimat_6mnd', 0))],
            'lege_overlevelsesestimat_2mnd': [float(features.get('lege_overlevelsesestimat_2mnd', 0))],
            'lege_overlevelsesestimat_6mnd': [float(features.get('lege_overlevelsesestimat_6mnd', 0))],
            'diabetes': [float(features.get('diabetes', 0))],
            'demens': [float(features.get('demens', 0))],
            'sykehusdød': [float(features.get('sykehusdød', 0))],
        })
        # Numeric data
        numeric_fields = [
            'alder', 'utdanning', 'blodtrykk', 'hvite_blodlegemer', 'hjertefrekvens',
            'respirasjonsfrekvens', 'kroppstemperatur', 'lungefunksjon', 'serumalbumin',
            'kreatinin', 'natrium', 'blod_ph', 'glukose', 'blodurea_nitrogen', 'urinmengde',
            'antall_komorbiditeter', 'koma_score', 'fysiologisk_score', 'apache_fysiologisk_score',
            'overlevelsesestimat_2mnd', 'overlevelsesestimat_6mnd',
            'lege_overlevelsesestimat_2mnd', 'lege_overlevelsesestimat_6mnd',
            'diabetes', 'demens', 'dødsfall'
        ]

        for field in numeric_fields:
            input_data[field] = [float(features.get(field, 0))]

        # Convert input_data into a pandas DataFrame
        input_df = pd.DataFrame(input_data)
        print(input_df)

        # Preprocess the input using the prepare_data_for_prediction function
        input_df_prepared, _, _, _ = prepare_data_for_prediction(input_df, prediction_mode=True)

        # Transform the input using the saved pipeline
        input_data_imputed = col_transformer.transform(input_df_prepared)
        
        # Create a DataFrame with feature names
        input_data_imputed_df = pd.DataFrame(input_data_imputed, columns=feature_names)
        print(input_data_imputed_df)

        # Predict the hospital stay length
        prediction = model.predict(input_data_imputed_df)[0]

        # Render the result on the webpage
        return render_template('index.html', prediction_text=f'Predicted Length of Stay: {prediction:.2f} days')

    except ValueError as e:
        print(f"Error processing input data: {e}")
        return render_template('index.html', prediction_text='Invalid input for one or more fields')

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
