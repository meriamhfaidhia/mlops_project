import sys
import os
import pytest

# Add the parent directory to the Python path so we can import model_pipeline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_pipeline import prepare_data  # Correct import for model_pipeline

def test_prepare_data():
    # Correct file paths for the CSV files in the root directory
    train_path = os.path.join(os.path.dirname(__file__), '..', 'churn-bigml-80.csv')  
    test_path = os.path.join(os.path.dirname(__file__), '..', 'churn-bigml-20.csv')

    # Check if the files exist
    assert os.path.exists(train_path), f"Train data file not found at {train_path}"
    assert os.path.exists(test_path), f"Test data file not found at {test_path}"

    # Call the prepare_data function
    X_train, X_test, y_train, y_test = prepare_data(train_path, test_path)
    
    # Check if the processed data files are created
    assert os.path.exists('train_data_prepared.csv'), "Train data file was not created"
    assert os.path.exists('test_data_prepared.csv'), "Test data file was not created"
    
    # Further assertions to check the shape and types of the returned data
    assert X_train.shape[0] == len(y_train), "Mismatch between X_train and y_train size"
    assert X_test.shape[0] == len(y_test), "Mismatch between X_test and y_test size"
    assert X_train.shape[1] == 13, "Incorrect number of features in X_train"
    assert X_test.shape[1] == 13, "Incorrect number of features in X_test"
