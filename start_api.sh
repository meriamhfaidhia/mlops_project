#!/bin/bash

set -e  # Stop the script in case of an error

# Define the project directory
PROJECT_DIR="/home/meriam/meriam-hfaidhia-4DS4-mlops_project"

cd "$PROJECT_DIR"

# Activate the virtual environment
. venv/bin/activate

# Check if a process is already running
if [ -f uvicorn_pid.txt ]; then
    PID=$(cat uvicorn_pid.txt)
    if ps -p $PID > /dev/null; then
        echo "⚠️ L'API tourne déjà avec le PID $PID."
        exit 1
    fi
fi

# Launch FastAPI with nohup and save the PID
nohup uvicorn app:app --host 0.0.0.0 --port 8000 --reload > uvicorn.log 2>&1 
echo $! > uvicorn_pid.txt

# Output the success message
echo "🚀 API démarrée avec succès. PID: $(cat uvicorn_pid.txt)"
echo "🌐 Accédez à Swagger : http://192.168.93.6:8000/docs"

