import argparse
import logging
import os
import pandas as pd

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model
)


def load_prepared_data(
    train_path="train_data_prepared.csv", test_path="test_data_prepared.csv"
):
    """Charge les données préparées si elles existent déjà."""
    if os.path.exists("train_data_prepared.csv") and os.path.exists(
        "test_data_prepared.csv"
    ):
        logging.info(
            "📂 Chargement des données préparées depuis les fichiers "
            "existants..."
        )

        X_train = pd.read_csv("train_data_prepared.csv").drop("Churn", axis=1)
        y_train = pd.read_csv("train_data_prepared.csv")["Churn"]
        X_test = pd.read_csv("test_data_prepared.csv").drop("Churn", axis=1)
        y_test = pd.read_csv("test_data_prepared.csv")["Churn"]

        return X_train, X_test, y_train, y_test
    return None, None, None, None


def main():
    """
    Pipeline de Machine Learning :
    - Prétraitement des données
    - Entraînement du modèle
    - Évaluation
    - Sauvegarde/Chargement
    - Prédiction
    """
    # Configurer le logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Parser des arguments
    parser = argparse.ArgumentParser(
        description="Pipeline de Machine Learning pour la prédiction du churn."
    )

    # Arguments du script
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
        "--train_path", type=str, required=True,
        help="Chemin du fichier CSV d'entraînement."
    )
    parser.add_argument(
        "--test_path", type=str, required=True,
        help="Chemin du fichier CSV de test."
    )

    # Analyser les arguments
    args = parser.parse_args()

    # Vérifier si les données préparées existent déjà
    X_train, X_test, y_train, y_test = load_prepared_data(
        args.train_path, args.test_path
    )

    # Si les données ne sont pas déjà préparées et qu'on doit les préparer
    if args.prepare or X_train is None:
        logging.info("🔄 Chargement et préparation des données...")
        X_train, X_test, y_train, y_test = prepare_data(
            args.train_path, args.test_path
        )

        logging.info("✅ Données préparées et sauvegardées.")
        if args.prepare:
            return  # Arrêter ici si on ne voulait que préparer les données

    model = None

    # Si un modèle doit être chargé
    if args.load:
        logging.info(f"📥 Chargement du modèle depuis {args.load}...")
        model = load_model(args.load)

    # Si le modèle doit être entraîné
    if args.train:
        logging.info("🚀 Entraînement du modèle...")
        model, params = train_model(X_train, y_train)

        if model:
            logging.info("✅ Modèle entraîné avec succès. Paramètres :")
            for param, value in params.items():
                logging.info(f"{param}: {value}")

    # Si un modèle est disponible, évaluation des performances
    if model and args.evaluate:
        logging.info("📊 Évaluation du modèle...")
        if not model:
            logging.info(
                "⚠️ Aucun modèle chargé ou entraîné. Veuillez entraîner un "
                "modèle avant l'évaluation."
            )
            return
        accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)

        result_message = (
            f"✅ Résultats de l'évaluation :\n"
            f"- Accuracy: {accuracy:.4f}\n"
            f"- Precision: {precision:.4f}\n"
            f"- Recall: {recall:.4f}\n"
            f"- F1-score: {f1:.4f}"
        )
        logging.info(result_message)
        print(result_message)

    # Si un modèle doit être sauvegardé
    if model and args.save:
        logging.info(f"💾 Sauvegarde du modèle dans {args.save}...")
        save_model(model, args.save)


if __name__ == "__main__":
    main
