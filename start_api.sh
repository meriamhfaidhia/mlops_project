#!/bin/bash

set -e  # Arrêter le script en cas d'erreur

# Définir le dossier du projet
PROJECT_DIR="/home/meriam/meriam-hfaidhia-4DS4-mlops_project"

cd "$PROJECT_DIR"

# Activer l'environnement virtuel
. venv/bin/activate

# Vérifier si un processus tourne déjà
if [ -f uvicorn_pid.txt ]; then
    PID=$(cat uvicorn_pid.txt)
    if ps -p $PID > /dev/null; then
        echo "⚠️ L'API tourne déjà avec le PID $PID."
        exit 1
    fi
fi

# Lancer FastAPI avec nohup et sauvegarder le PID
uvicorn app:app --host 0.0.0.0 --port 8000 --reload > uvicorn.log 2>&1&

echo "🚀 API démarrée avec succès. PID: $(cat uvicorn_pid.txt)"
echo "🌐 Accédez à Swagger : http://192.168.93.6:8000/docs"

