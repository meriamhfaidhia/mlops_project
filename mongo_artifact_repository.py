from pymongo import MongoClient
import os
import joblib
import pandas as pd
import logging

class MongoArtifactRepository:
    def __init__(self, mongo_uri):
        """
        Initialise la connexion à MongoDB.
        """
        self.client = MongoClient(mongo_uri)
        self.db = self.client.get_database()  # Utilisation de la base de données par défaut
        self.artifacts_collection = self.db.get_collection("artifacts")  # Collection dédiée aux artefacts
        logging.info("✅ Connexion à MongoDB établie.")

    def log_artifact(self, local_path, artifact_path):
        """
        Enregistre un artefact dans MongoDB en sérialisant le fichier local (par exemple, un modèle).
        """
        try:
            # Vérifier si le fichier existe
            if os.path.exists(local_path):
                # Charger et sérialiser le modèle (ou artefact)
                with open(local_path, "rb") as f:
                    artifact_data = f.read()  # Lire le fichier en binaire

                # Enregistrer l'artefact dans MongoDB
                self.artifacts_collection.insert_one({
                    "artifact_path": artifact_path,
                    "artifact_data": artifact_data,
                    "created_at": pd.to_datetime("now")
                })

                logging.info(f"✅ Artefact {artifact_path} enregistré dans MongoDB.")
            else:
                logging.error(f"❌ Le fichier local {local_path} n'existe pas.")
        except Exception as e:
            logging.error(f"❌ Erreur lors de l'enregistrement de l'artefact dans MongoDB : {e}")


    def get_artifact(self, artifact_path):
          artifact = self.db.artifacts.find_one({"artifact_path": artifact_path})
          return artifact["artifact_data"] if artifact else None
