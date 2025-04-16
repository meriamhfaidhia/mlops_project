from flask import Flask, request, render_template
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Charger le modèle
model = joblib.load('prediction_model.joblib')

@app.route('/dashboards')
def dashboards():
    # Données pour les graphiques
    dashboard_data = {
        'customer_health': {
            'title': 'Santé Clients',
            'charts': {
                'churn_risk': {
                    'type': 'doughnut',
                    'data': [32, 68],  # 32% risque, 68% safe
                    'labels': ['À risque', 'Fidèles'],
                    'colors': ['#ff6384', '#36a2eb']
                },
                'satisfaction': {
                    'type': 'bar',
                    'data': [15, 35, 40, 10],  # Très satisfait à très insatisfait
                    'labels': ['Très satisfait', 'Satisfait', 'Neutre', 'Insatisfait'],
                    'colors': ['#4bc0c0', '#4bc0c0', '#ffcd56', '#ff6384']
                }
            }
        },
        'retention': {
            'title': 'Rétention Clients',
            'charts': {
                'monthly_churn': {
                    'type': 'line',
                    'data': [12, 19, 15, 8, 7, 10],
                    'labels': ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin'],
                    'colors': '#9966ff'
                },
                'retention_reasons': {
                    'type': 'polarArea',
                    'data': [35, 25, 20, 15, 5],
                    'labels': ['Prix', 'Service', 'Produit', 'Concurrence', 'Autre'],
                    'colors': ['#ff9f40', '#ff6384', '#36a2eb', '#9966ff', '#c9cbcf']
                }
            }
        }
    }
    return render_template('dashboards.html', dashboards=dashboard_data)

@app.route('/retention-tips')
def retention_tips():
    return render_template('retention_tips.html')

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

         # Effectuer la prédiction
        prediction = model.predict(input_data)[0]
        # Convertir la prédiction en un type Python natif (par exemple, int)
        prediction_result = "Churn" if prediction[0] == 1 else "Not Churn"
        print(f"Résultat de la prédiction : {prediction}")
        return render_template('result.html', result=prediction)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)


