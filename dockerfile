# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the rest of the application code
COPY . .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir -r requierements.txt

# Create the mlruns directory and set permissions
RUN mkdir -p /app/mlruns && \
    chown -R 1000:1000 /app/mlruns

# Expose the port for the MLflow server
EXPOSE 5000

# Default command to start the MLflow UI
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--backend-store-uri", "sqlite:////app/mlflow.db", "--default-artifact-root", "file:///app/mlruns"]
