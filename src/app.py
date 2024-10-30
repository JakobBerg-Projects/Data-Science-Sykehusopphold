import numpy as np
import pandas as pd  
import pickle
from flask import Flask, request, render_template
from waitress import serve
from preprocessing import prepare_data_for_length_prediction, prepare_data_for_death_classification


app = Flask(__name__)

# Last inn den trente regresjonsmodellen fra 'model.pkl'
model, col_transformer, feature_names= pickle.load(open('model.pkl', 'rb'))

# Last inn den trente klassifikasjonsmodellen fra 'sykehusdod_model.pkl'
sykehusdod_model = pickle.load(open('sykehusdod_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Hent data fra skjemaet
    features = dict(request.form)
    
    try:
        # Henter ut alle inputverdier fra skjemaet
        input_data = {}
        
        # Kategoriske data
        sub_disease = features.get('sykdom_underkategori', 'other')
        input_data.update({
            'kjønn': [features.get('kjønn', 'male')],  # Default to 'male'
            'etnisitet': [features.get('etnisitet', 'white')],  # Default to 'white'
            'sykdom_underkategori': [sub_disease],  # Use the selected sub-disease
            'dnr_status': [features.get('dnr_status', 'no')],  # Default to 'no'
            'inntekt': [features.get('inntekt', 'unknown')],  # Default to 'unknown'
            'kreft': [features.get('kreft', 'no')],  # Default to 'no'
            'sykehusdød': [0]  # Set 'dødsfall' to 0 by default
        })

        # Automatiser tildeling av sykdomskategori basert på sykdom_underkategori
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

       
        # Numeriske data
        numeric_fields = [
            'alder', 'utdanning', 'blodtrykk', 'hvite_blodlegemer', 'hjertefrekvens',
            'respirasjonsfrekvens', 'kroppstemperatur', 'lungefunksjon', 'serumalbumin',
            'kreatinin', 'natrium', 'blod_ph', 'glukose', 'blodurea_nitrogen', 'urinmengde',
            'antall_komorbiditeter', 'koma_score', 'fysiologisk_score', 'apache_fysiologisk_score',
            'overlevelsesestimat_2mnd', 'overlevelsesestimat_6mnd',
            'lege_overlevelsesestimat_2mnd', 'lege_overlevelsesestimat_6mnd',
            'diabetes', 'demens', 'dødsfall'
        ]

        # Legg til numeriske felt i input_data med NaN for tomme felt
        for field in numeric_fields:
            value = features.get(field, 0)
            input_data[field] = [float(value) if value != "" else np.nan]

        # Konverter input_data til en pandas DataFrame
        input_df = pd.DataFrame(input_data)
        input_df.replace('', np.nan, inplace=True) # Erstatt tomme strenger med NaN
        

        # Imputer `sykehusdød` ved hjelp av klassifikasjonsmodellen
        X_classification, _, _, _ = prepare_data_for_death_classification(input_df, prediction_mode=True)
        sykehusdod_prediction = sykehusdod_model.predict(X_classification)[0]
        input_df['sykehusdød'] = sykehusdod_prediction

        # Forbered data til lengdeprediksjon
        input_df_prepared, _, _, _ = prepare_data_for_length_prediction(input_df, prediction_mode=True)

        # Transformer input data ved hjelp av `col_transformer`
        input_data_imputed = col_transformer.transform(input_df_prepared)
        
        # Konverter transformert data til DataFrame med riktige kolonnenavn
        input_data_imputed_df = pd.DataFrame(input_data_imputed, columns=feature_names)

        # Prediker lengden på oppholdet
        prediction = model.predict(input_data_imputed_df)[0]

        # Vis en risikomelding for død hvis `sykehusdød` er positiv
        death_risk_message = "Advarsel: Høy risiko for død." if sykehusdod_prediction == 1 else None

        # Vis resultatet på nettsiden
        return render_template(
            'index.html', 
            prediction_text=f'Predicted Length of Stay: {prediction:.2f} days',
            death_risk_message=death_risk_message
        )

    except ValueError as e:
        print(f"Error processing input data: {e}")
        

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
