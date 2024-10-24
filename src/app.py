import numpy as np
import pandas as pd  # Import pandas for creating DataFrame
import pickle
from flask import Flask, request, render_template
from waitress import serve
from imputering_og_modellering import prepare_data_for_prediction, col_transformer
import importnb
from imputering_og_modellering import prepare_data_for_prediction, col_transformer


app = Flask(__name__)

# Load the trained model from 'model.pkl'
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "Hello, Flask!"

@app.route('/predict', methods=['POST'])
def predict():
    ''' 
    Render the prediction result on HTML
    '''
    # Get data from form
    features = dict(request.form)
    
    try:
        # Extract all input values from the form
        input_data = {
            'alder': [float(features.get('alder', 0))],
            'kjonn': [features.get('kjonn', 'male')],
            'utdanning': [float(features.get('utdanning', 0))],
            'inntekt': [float(features.get('inntekt', 0))],
            'etnisitet': [features.get('etnisitet', 'white')],
            'sykehusdod': [float(features.get('sykehusdod', 0))],
            'blodtrykk': [float(features.get('blodtrykk', 0))],
            'hvite_blodlegemer': [float(features.get('hvite_blodlegemer', 0))],
            'hjertefrekvens': [float(features.get('hjertefrekvens', 0))],
            'respirasjonsfrekvens': [float(features.get('respirasjonsfrekvens', 0))],
            'kroppstemperatur': [float(features.get('kroppstemperatur', 0))],
            'lungefunksjon': [float(features.get('lungefunksjon', 0))],
            'serumalbumin': [float(features.get('serumalbumin', 0))],
            'bilirubin': [float(features.get('bilirubin', 0))],
            'kreatinin': [float(features.get('kreatinin', 0))],
            'natrium': [float(features.get('natrium', 0))],
            'blod_ph': [float(features.get('blod_ph', 0))],
            'glukose': [float(features.get('glukose', 0))],
            'blodurea_nitrogen': [float(features.get('blodurea_nitrogen', 0))],
            'urinmengde': [float(features.get('urinmengde', 0))],
            'sykdomskategori_id': [float(features.get('sykdomskategori_id', 0))],
            'sykdomskategori': [features.get('sykdomskategori', 'other')],
            'dodsfall': [float(features.get('dodsfall', 0))],
            'sykdom_underkategori': [features.get('sykdom_underkategori', 'other')],
            'antall_komorbiditeter': [float(features.get('antall_komorbiditeter', 0))],
            'koma_score': [float(features.get('koma_score', 0))],
            'adl_pasient': [float(features.get('adl_pasient', 0))],
            'adl_stedfortreder': [float(features.get('adl_stedfortreder', 0))],
            'fysiologisk_score': [float(features.get('fysiologisk_score', 0))],
            'apache_fysiologisk_score': [float(features.get('apache_fysiologisk_score', 0))],
            'overlevelsesestimat_2mnd': [float(features.get('overlevelsesestimat_2mnd', 0))],
            'overlevelsesestimat_6mnd': [float(features.get('overlevelsesestimat_6mnd', 0))],
            'diabetes': [float(features.get('diabetes', 0))],
            'demens': [float(features.get('demens', 0))],
            'kreft': [float(features.get('kreft', 0))],
            'lege_overlevelsesestimat_2mnd': [float(features.get('lege_overlevelsesestimat_2mnd', 0))],
            'lege_overlevelsesestimat_6mnd': [float(features.get('lege_overlevelsesestimat_6mnd', 0))],
            'dnr_status': [features.get('dnr_status', 'no')]
        }

        # Convert input_data into a pandas DataFrame
        input_df = pd.DataFrame(input_data)

        # Preprocess the input using the prepare_data_for_prediction function
        input_df_prepared, numeric_cols_pred, categorical_cols_pred = prepare_data_for_prediction(
            input_df, categorical_cols_train, numeric_cols_train
        )

        # Transform the input using the saved pipeline
        input_data_imputed = col_transformer.transform(input_df_prepared)

        # Predict the hospital stay length
        prediction = model.predict(input_data_imputed)[0]

        return render_template('index.html', prediction_text=f'Predicted Length of Stay: {prediction:.2f} days')

    except ValueError:
        return render_template('index.html', prediction_text='Invalid input for one or more fields')

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
