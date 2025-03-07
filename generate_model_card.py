import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import os
import logging

def load_prepared_data(train_path="train_data_prepared.csv", test_path="test_data_prepared.csv"):
    if os.path.exists(train_path) and os.path.exists(test_path):
        logging.info(
            "📂 Chargement des données préparées depuis les fichiers existants..."
        )
        X_train = pd.read_csv(train_path).drop("Churn", axis=1)
        y_train = pd.read_csv(train_path)["Churn"]
        X_test = pd.read_csv(test_path).drop("Churn", axis=1)
        y_test = pd.read_csv(test_path)["Churn"]
        return X_train, X_test, y_train, y_test
    return None, None, None, None

def generate_model_card(model_path, output_path, test_path="test_data_prepared.csv"):
    model = joblib.load(model_path)
    
    # Charger les données de test
    _, X_test, _, y_true = load_prepared_data(test_path=test_path)
    
    # Faire des prédictions
    y_pred = model.predict(X_test)
    
    # Calculer les métriques
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    # Générer la fiche de modèle
    with open(output_path, 'w') as f:
        f.write("# Fiche de Modèle\n\n")
        f.write("## Hyperparamètres\n")
        f.write(str(model.get_params()) + "\n\n")
        f.write("## Métriques de Performance\n")
        f.write(pd.DataFrame.from_dict(metrics, orient='index').to_markdown())
    
    print(f"Fiche de modèle générée : {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, default="test_data_prepared.csv")
    args = parser.parse_args()
    
    generate_model_card(args.model_path, args.output_path, args.test_path)