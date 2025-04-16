#!/bin/bash

set -e  # Arrêter le script en cas d'erreur

# Définir le dossier du projet
PROJECT_DIR="/home/meriam/meriam-hfidhia-4DS4-mlops_project"

cd "$PROJECT_DIR"

# Vérifier si le fichier PID existe
if [ -f uvicorn_pid.txt ]; then
    PID=$(cat uvicorn_pid.txt)

    # Vérifier si le processus est en cours d'exécution
    if ps -p $PID > /dev/null; then
        echo "🚦 Arrêt de l'API avec le PID $PID..."

        # Tuer le processus uvicorn
        kill $PID
        echo "✅ API arrêtée avec succès."
    else
        echo "⚠️ Aucun processus API trouvé pour le PID $PID."
    fi

    # Supprimer le fichier PID
    rm uvicorn_pid.txt
else
    echo "⚠️ Le fichier uvicorn_pid.txt est introuvable. L'API n'est pas démarrée."
fi

