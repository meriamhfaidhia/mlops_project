import subprocess
import logging
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from pymongo import MongoClient
import argparse
from contextlib import nullcontext
from mongo_artifact_repository import MongoArtifactRepository
from elasticsearch import Elasticsearch


from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model
)


# Fonction pour vérifier et démarrer MongoDB si nécessaire

def check_and_start_mongodb():
    """
    Vérifie si MongoDB est en cours d'exécution dans Docker, sinon le démarre.
    """
    try:
        # Vérifier si MongoDB dans Docker est en cours d'exécution
        result = subprocess.run(
            ['docker', 'ps', '-q', '-f', 'name=mongodb_official'],
            check=True,
            capture_output=True
        )
        if result.stdout.strip():
            logging.info(
                "✅ MongoDB est déjà en cours d'exécution dans Docker."
                )
        else:
            logging.info(
                "❌ MongoDB n'est pas en cours d'exécution dans Docker. "
                "Tentative de démarrage..."
            )
            subprocess.run(['docker', 'start', 'mongodb_official'], check=True)
            logging.info("✅ MongoDB a été démarré dans Docker.")
    except subprocess.CalledProcessError as e:
        logging.error(
            f"Erreur lors de la vérification/démarrage de MongoDB : {str(e)}"
            )
        exit(1)


def connect_to_elasticsearch():
    """
    Connexion à Elasticsearch.
    """
    try:
        es = Elasticsearch(
            "http://localhost:9200",
            verify_certs=False
            )
        if es.info():
            logging.info("✅ Connexion à Elasticsearch réussie.")
            return es
        else:
            logging.error("❌ Impossible de se connecter à Elasticsearch.")
            return None
    except Exception as e:
        logging.error(f"❌ Erreur lors de la connexion à Elasticsearch : {e}")
        return None


def send_logs_to_elasticsearch(es, run_id, params, metrics):
    """
    Envoie les logs de MLflow (paramètres et métriques) à Elasticsearch.
    """
    if es is None:
        logging.warning("❌ Elasticsearch n'est pas connecté.")
        return

    log_data = {
        "run_id": run_id,
        "params": params,
        "metrics": metrics,
        "timestamp": pd.Timestamp.now().isoformat()
    }

    try:
        es.index(index="mlflow-metrics", body=log_data)
        logging.info("✅ Logs envoyés à Elasticsearch avec succès.")
    except Exception as e:
        logging.error(f"❌ Erreur lors de l'envoi des logs : {e}")


# Fonction pour vérifier la connexion à MongoDB
def check_mongodb_connection(mongo_uri):
    try:
        client = MongoClient(mongo_uri)
        client.server_info()  # Vérifie la connexion
        logging.info("✅ Connexion à MongoDB réussie.")
    except Exception as e:
        logging.error(f"Erreur lors de la connexion à MongoDB : {e}")
        exit(1)


def configure_mlflow(experiment_name="churn"):
    # Define URIs
    sqlite_uri = "sqlite:///mlflow.db"
    mongo_uri = "mongodb://172.17.0.2:27017/mlflow_artifacts"
    os.environ["MLFLOW_TRACKING_URI"] = sqlite_uri
    os.environ["MLFLOW_ARTIFACT_URI"] = mongo_uri
    # Check MongoDB connection
    check_mongodb_connection(mongo_uri)
    try:
        # Set the tracking URI
        mlflow.set_tracking_uri(sqlite_uri)
        logging.info(
            f"✅ MLflow tracking URI set to: {sqlite_uri}"
            )

        # Check if the experiment exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None:
            # If the experiment exists, log its details
            experiment_id = experiment.experiment_id
            logging.info(
                f"✅ Expérience '{experiment_name}' existe déjà."
                f" ID: {experiment_id}"
            )
        else:
            # Create the experiment if it doesn't exist
            mlflow.create_experiment(
                experiment_name,
                artifact_location=mongo_uri
                )
            logging.info(
                f"✅ Expérience '{experiment_name}' créée avec succès."
                f"Artifact location: {mongo_uri}"
             )

        # Set the current experiment
        mlflow.set_experiment(experiment_name)
        logging.info(f"✅ Expérience actuelle définie sur '{experiment_name}'.")
    except Exception as e:
        logging.error(
            f"❌ Erreur lors de la configuration de MLflow : {str(e)}"
            )
        raise  # Re-raise the exception to see the full traceback


# Enregistrement des artefacts dans MongoDB et MLflow
def log_artifacts_to_mongo_and_mlflow(model, model_name="prediction_model"):
    try:
        model_path = f"{model_name}.joblib"
        joblib.dump(model, model_path)  # Sauvegarder le modèle dans un fichier
        logging.info(
            f"✅ Modèle sauvegardé localement sous {model_path}."
            )

        # Connexion à MongoDB et enregistrement du modèle
        mongo_uri = "mongodb://172.17.0.2:27017/mlflow_artifacts"
        artifact_repo = MongoArtifactRepository(mongo_uri)
        # Log du modèle dans MongoDB
        artifact_repo.log_artifact(
            local_path=model_path,
            artifact_path=f"{model_name}/prediction_model.joblib"
            )
        logging.info(
            f"✅ Modèle {model_name} enregistré dans MongoDB."
            )
        # Enregistrer le modèle dans MLflow
        mlflow.log_artifact(model_path)
        logging.info(f"✅ Modèle {model_name} enregistré dans MLflow.")
    except Exception as e:
        logging.info(
            f"❌ Erreur lors de l'enregistrement des artefacts "
            f"dans MongoDB et MLflow : {e}"
             )


# Charger les données préparées si elles existent déjà
def load_prepared_data(
    train_path="train_data_prepared.csv", test_path="test_data_prepared.csv"
):
    if os.path.exists(train_path) and os.path.exists(test_path):
        logging.info(
            "📂 Chargement des données préparées"
            " depuis les fichiers existants..."
        )
        X_train = pd.read_csv(train_path).drop("Churn", axis=1)
        y_train = pd.read_csv(train_path)["Churn"]
        X_test = pd.read_csv(test_path).drop("Churn", axis=1)
        y_test = pd.read_csv(test_path)["Churn"]
        return X_train, X_test, y_train, y_test
    return None, None, None, None


def promote_model(model_name, version, stage):
    client = MlflowClient()
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        logging.info(
            f"✅ Modèle {model_name} version {version} promu au stage {stage}."
            )
    except Exception as e:
        logging.error(
            f"Erreur lors de la promotion du modèle : {str(e)}"
            )


def register_model(
    model,
    model_name="churn_model",
    initial_stage="Staging",
    X_train=None
):
    client = MlflowClient()  # noqa: F841

    model_uri = f"runs:/{mlflow.active_run().info.run_id}/churn_model"
    try:
        if X_train is not None:
            signature = infer_signature(
                X_train,
                model.predict(X_train))  # noqa: F841
        else:
            signature = None  # noqa: F841

        model_version = mlflow.register_model(model_uri, model_name)
        logging.info(
            f"Modèle {model_name} enregistré avec la version "
            f"{model_version.version}."
        )

        # ✅ Cette ligne doit être à l'intérieur du bloc try
        promote_model(model_name, model_version.version, initial_stage)
    except Exception as e:
        logging.error(
            f"Erreur lors de l'enregistrement du modèle : {str(e)}"
        )
        return None

    return model_version


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Pipeline de Machine Learning pour la prédiction du churn."
    )

    parser.add_argument(
        "--prepare", action="store_true", help="Préparer les données."
    )
    parser.add_argument(
        "--train", action="store_true", help="Entraîner le modèle."
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Évaluer le modèle."
    )
    parser.add_argument(
        "--save", type=str, help="Sauvegarder le modèle dans un fichier."
    )
    parser.add_argument(
        "--load", type=str, help="Charger un modèle existant."
    )
    parser.add_argument(
        "--train_path", type=str, help="Chemin du fichier CSV d'entraînement."
    )
    parser.add_argument(
        "--test_path", type=str, help="Chemin du fichier CSV de test."
    )
    parser.add_argument(
        "--promote", type=str, choices=["Staging", "Production"],
        help=(
            "Promouvoir un modèle à un stage spécifique"
            "(Staging ou Production).")
    )
    parser.add_argument(
        "--model_version", type=int,
        help="Version du modèle à promouvoir (requis avec --promote)."
    )

    args = parser.parse_args()

    if not args.evaluate and not args.save:
        check_and_start_mongodb()
        configure_mlflow()
        mlflow.set_experiment("churn")

    if args.promote:
        if not args.model_version:
            logging.error(
                "❌ L'argument --model_version"
                "est requis pour promouvoir un modèle."
                )
            return
        promote_model("churn_model", args.model_version, args.promote)
        return

    if not args.train_path or not args.test_path:
        logging.error(
            "❌ Les arguments --train_path et --test_path"
            " sont requis pour l'entraînement,"
            " l'évaluation ou la préparation des données."
        )
        return

    X_train, X_test, y_train, y_test = load_prepared_data(
        args.train_path, args.test_path
    )

    if args.prepare or X_train is None:
        logging.info(
            "🔄 Chargement et préparation des données..."
            )
        X_train, X_test, y_train, y_test = prepare_data(
            args.train_path, args.test_path
        )
        logging.info(
            "✅ Données préparées et sauvegardées."
            )
        if args.prepare:
            return

    model = None
    with (
        mlflow.start_run()
        if not args.evaluate and not args.save
        else nullcontext()
    ):
        if args.load:
            logging.info(
                f"📥 Chargement du modèle depuis {args.load}..."
                )
            model = load_model(args.load)

        if args.train:
            logging.info(
                "🚀 Entraînement du modèle..."
                )
            model, params = train_model(X_train, y_train)

            if model:
                logging.info(
                    "✅ Modèle entraîné avec succès. Paramètres :"
                    )
                for param, value in params.items():
                    logging.info(f"{param}: {value}")
                    mlflow.log_param(param, value)

                signature = infer_signature(X_train, y_train)

                mlflow.sklearn.log_model(
                    model,
                    "churn_model",
                    signature=signature)

                model_version = register_model(model, X_train=X_train)
                if model_version:
                    logging.info(
                        f"Le modèle a été enregistré avec "
                        f"la version {model_version.version}."
                        )

                accuracy, precision, recall, f1 = (
                    evaluate_model(model, X_test, y_test)
                    )

                result_message = (
                    f"✅ Résultats de l'entraînement :\n"
                    f"- Accuracy: {accuracy:.4f}\n"
                    f"- Precision: {precision:.4f}\n"
                    f"- Recall: {recall:.4f}\n"
                    f"- F1-score: {f1:.4f}"
                )
                logging.info(result_message)
                print(result_message)

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                es = connect_to_elasticsearch()
                if es:
                    metrics = {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1
                    }
                    send_logs_to_elasticsearch(
                        es, mlflow.active_run().info.run_id, params, metrics
                        )

                logging.info(
                    "✅ Enregistrement des artefacts dans MongoDB et MLflow...")
                log_artifacts_to_mongo_and_mlflow(model)

        if model and args.evaluate:
            logging.info("📊 Évaluation du modèle...")
            accuracy, precision, recall, f1 = (
                evaluate_model(model, X_test, y_test)
                )

            result_message = (
                f"✅ Résultats de l'évaluation :\n"
                f"- Accuracy: {accuracy:.4f}\n"
                f"- Precision: {precision:.4f}\n"
                f"- Recall: {recall:.4f}\n"
                f"- F1-score: {f1:.4f}"
            )
            logging.info(result_message)

        if model and args.save:
            logging.info(f"💾 Sauvegarde du modèle dans {args.save}...")
            save_model(model, args.save)


if __name__ == "__main__":
    main()
