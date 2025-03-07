from flask import Flask, request, render_template
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Charger le modèle
model = joblib.load('prediction_model.joblib')

# Définir la route pour l'index
@app.route('/')
def index():
    return render_template('index.html')

# Route pour faire la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les valeurs du formulaire
        account_length = int(request.form['account_length'])
        number_vmail_messages = int(request.form['number_vmail_messages'])
        total_day_calls = int(request.form['total_day_calls'])
        total_day_charge = float(request.form['total_day_charge'])
        total_eve_calls = int(request.form['total_eve_calls'])
        total_eve_charge = float(request.form['total_eve_charge'])
        total_night_calls = int(request.form['total_night_calls'])
        total_night_charge = float(request.form['total_night_charge'])
        total_intl_calls = int(request.form['total_intl_calls'])
        total_intl_charge = float(request.form['total_intl_charge'])
        customer_service_calls = int(request.form['customer_service_calls'])

        # Encodage des variables catégorielles
        international_plan = int(request.form['international_plan'])
        voice_mail_plan = int(request.form['voice_mail_plan'])

        # Créer un tableau numpy avec les données d'entrée
        input_data = np.array([[account_length, number_vmail_messages, total_day_calls, total_day_charge,
                                total_eve_calls, total_eve_charge, total_night_calls, total_night_charge,
                                total_intl_calls, total_intl_charge, customer_service_calls,
                                international_plan, voice_mail_plan]])

        # Faire la prédiction
        prediction = model.predict(input_data)

        # Convertir la prédiction en un type Python natif (par exemple, int)
        prediction_result = "Churn" if prediction[0] == 1 else "Not Churn"

        return render_template('result.html', prediction=prediction_result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)


