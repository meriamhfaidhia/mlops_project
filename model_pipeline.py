import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import joblib
from scipy.stats import zscore


def normalize_data(data, columns_to_normalize):
    """
    Normaliser les données avec MinMaxScaler.
    """
    scaler = MinMaxScaler()
    data[columns_to_normalize] = scaler.fit_transform(
        data[columns_to_normalize]
    )
    return data


def drop_columns(data, columns_to_drop):
    """
    Supprimer les colonnes inutiles.
    """
    return data.drop(columns=columns_to_drop)


def remove_outliers(data, num_cols, method="zscore", threshold=3):
    """
    Supprimer les outliers en utilisant la méthode choisie (zscore ou iqr).
    """
    if method == "zscore":
        z_scores = np.abs(zscore(data[num_cols]))
        data = data[(z_scores < threshold).all(axis=1)]
    elif method == "iqr":
        for col in num_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = data[
                (data[col] >= lower_bound) & (data[col] <= upper_bound)
            ]
    return data


def prepare_data(train_path, test_path):
    """
    Charger et préparer les données d'entraînement et de test.
    """
    # Charger les données
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Fusionner les données pour prétraitement
    data = pd.concat([train_data, test_data], ignore_index=True)

    # Supprimer les colonnes inutiles
    columns_to_drop = [
        'State', 'Area code', 'Total day minutes', 'Total eve minutes',
        'Total night minutes', 'Total intl minutes'
    ]
    data = drop_columns(data, columns_to_drop)

    # Encodage des variables catégorielles
    label_encoder = LabelEncoder()
    data['International plan'] = label_encoder.fit_transform(
        data['International plan']
    )
    data['Voice mail plan'] = label_encoder.fit_transform(
        data['Voice mail plan']
    )
    data['Churn'] = label_encoder.fit_transform(data['Churn'])

    # Normalisation des données numériques
    numerical_columns = [
        'Account length', 'Number vmail messages', 'Total day calls',
        'Total day charge', 'Total eve calls', 'Total eve charge',
        'Total night calls', 'Total night charge', 'Total intl calls',
        'Total intl charge', 'Customer service calls'
    ]

    # Suppression des outliers
    data = remove_outliers(data, numerical_columns, method="iqr")

    # Normalisation des colonnes numériques
    data = normalize_data(data, numerical_columns)

    # Séparer les données en ensembles d'entraînement et de test
    train_data = data.iloc[:len(train_data)]
    test_data = data.iloc[len(train_data):]

    # Séparer les caractéristiques et la cible
    X_train = train_data.drop('Churn', axis=1)
    y_train = train_data['Churn']
    X_test = test_data.drop('Churn', axis=1)
    y_test = test_data['Churn']

    # Sauvegarder les résultats dans des fichiers CSV
    train_data.to_csv("train_data_prepared.csv", index=False)
    test_data.to_csv("test_data_prepared.csv", index=False)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Entraîner le modèle RandomForest et obtenir ses paramètres.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    params = model.get_params()
    return model, params


def evaluate_model(model, X_test, y_test):
    """
    Évaluer le modèle en calculant les métriques de performance.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1


def save_model(model, filename):
    """
    Sauvegarder le modèle dans un fichier.
    """
    joblib.dump(model, filename)


def load_model(filename):
    """
    Charger un modèle depuis un fichier.
    """
    return joblib.load(filename)
