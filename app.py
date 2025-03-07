from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model_pipeline import prepare_data
from pydantic import BaseModel
import joblib
import numpy as np
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Configuration de la base de données
DATABASE_URL = "postgresql://postgres:wael@db:5432/mlops_db"

# Créer une instance de moteur SQLAlchemy
engine = create_engine(DATABASE_URL)

# Créer une session pour interagir avec la base de données
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Déclarer une base pour les modèles SQLAlchemy
Base = declarative_base()



# Définir un modèle SQLAlchemy pour la table des prédictions
class PredictionRecord(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    features = Column(String)  # Stocker les features sous forme de chaîne JSON
    prediction = Column(Integer)  # Résultat de la prédiction
    timestamp = Column(DateTime, default=datetime.utcnow)  # Horodatage de la prédiction

# Créer la table dans la base de données (si elle n'existe pas déjà)
Base.metadata.create_all(bind=engine)

# Créer une instance FastAPI
app = FastAPI()

# Chemin du modèle
MODEL_PATH = "prediction_model.joblib"

# Vérifier si le modèle existe
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Le fichier du modèle est introuvable à l'emplacement: {MODEL_PATH}")

# Charger le modèle préalablement sauvegardé
model = joblib.load(MODEL_PATH)

# Définir un modèle Pydantic pour valider les données d'entrée
class PredictionRequest(BaseModel):
    features: list  # Assurez-vous que les données sont des float et contiennent au moins une caractéristique

class RetrainRequest(BaseModel):
    n_estimators: int
    max_depth: int
    min_samples_split: int
    train_path: str
    test_path: str

# Définir une route HTTP POST pour les prédictions
@app.post("/predict/")
def predict(request: PredictionRequest):
    try:
        # Convertir les données d'entrée en un tableau numpy
        input_data = np.array(request.features).reshape(1, -1)

        # Log the shape of the input data
        print(f"Input data shape: {input_data.shape}")

        # Utiliser le modèle pour prédire la sortie
        prediction = model.predict(input_data)

        # Log the raw prediction result
        print(f"Prediction (raw): {prediction}")

        # Convertir la prédiction en un type Python natif (par exemple, int)
        prediction_result = int(prediction[0])

        # Enregistrer la prédiction dans la base de données
        db = SessionLocal()
        prediction_record = PredictionRecord(
            features=str(request.features),  # Convertir la liste en chaîne JSON
            prediction=prediction_result
        )
        db.add(prediction_record)
        db.commit()
        db.refresh(prediction_record)
        db.close()

        return {"prediction": prediction_result}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Erreur de valeur des données d'entrée: {ve}")
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=f"Fichier du modèle introuvable: {fnf}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {e}")

# Route pour réentraîner le modèle avec de nouveaux hyperparamètres
@app.post("/retrain/")
def retrain(request: RetrainRequest):
    try:
        # Préparer les données d'entraînement et de test
        X_train, X_test, y_train, y_test = prepare_data(request.train_path, request.test_path)

        # Créer un nouveau modèle avec les paramètres fournis
        new_model = RandomForestClassifier(
            n_estimators=request.n_estimators,
            max_depth=request.max_depth,
            min_samples_split=request.min_samples_split
        )

        # Entraîner le modèle
        new_model.fit(X_train, y_train)

        # Évaluer le modèle sur les données de test
        y_pred = new_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Sauvegarder le modèle réentraîné
        joblib.dump(new_model, MODEL_PATH)

        return {"message": "Modèle réentraîné avec succès", "accuracy": accuracy}

    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=f"Fichier de données introuvable: {fnf}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du réentraînement du modèle : {e}")
        

