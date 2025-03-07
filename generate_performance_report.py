import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
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

def generate_performance_report(model_path, output_path, test_path="test_data_prepared.csv"):
    # Charger le modèle
    model = joblib.load(model_path)
    
    # Charger les données de test
    _, X_test, _, y_test = load_prepared_data(test_path=test_path)
    
    # Faire des prédictions
    y_pred = model.predict(X_test)
    
    # Générer la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    # Générer le rapport de classification
    report = classification_report(y_test, y_pred)
    
    # Sauvegarder la matrice de confusion sous forme d'image
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    confusion_matrix_path = f"{output_path}_confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    # Générer le rapport de performance
    with open(output_path, 'w') as f:
        f.write("# Rapport de Performance\n\n")
        f.write("## Matrice de Confusion\n")
        f.write(f"![Confusion Matrix]({confusion_matrix_path})\n\n")
        f.write("## Rapport de Classification\n")
        f.write(report)
    
    print(f"Rapport de performance généré : {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, default="test_data_prepared.csv")
    args = parser.parse_args()
    
    generate_performance_report(args.model_path, args.output_path, args.test_path)