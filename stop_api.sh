#!/bin/bash

set -e  # Stop the script in case of an error

# Define the project directory
PROJECT_DIR="/home/meriam/meriam-hfaidhia-4DS4-mlops_project"

cd "$PROJECT_DIR"

# Check if the PID file exists
if [ ! -f uvicorn_pid.txt ]; then
    echo "⚠️ Aucun fichier PID trouvé. L'API n'est probablement pas lancée."
    exit 1
fi

# Read the PID from the file
PID=$(cat uvicorn_pid.txt)

# Check if the process is running
if ps -p $PID > /dev/null; then
    # Stop the FastAPI server
    kill $PID
    echo "🛑 API stoppée avec succès. PID: $PID"
    # Remove the PID file
    rm uvicorn_pid.txt
else
    echo "⚠️ Aucune API en cours d'exécution avec le PID $PID."
    exit 1
fi

