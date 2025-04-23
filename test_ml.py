import pytest
# TODO: add necessary import
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, inference


# Sample data for testing
@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        "age": [25, 30],
        "workclass": ["Private", "Self-emp-not-inc"],
        "fnlgt": [226802, 89814],
        "education": ["11th", "HS-grad"],
        "education-num": [7, 9],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Machine-op-inspct", "Farming-fishing"],
        "relationship": ["Own-child", "Husband"],
        "race": ["Black", "White"],
        "sex": ["Male", "Male"],
        "capital-gain": [0, 0],
        "capital-loss": [0, 0],
        "hours-per-week": [40, 50],
        "native-country": ["United-States", "United-States"],
        "salary": ["<=50K", ">50K"]
    })
    return data

# TODO: implement the first test. Change the function name and input as needed
def test_process_data(sample_data):
    """
    Tests the process_data function to ensure it correctly processes the input data.

    This test verifies that:
    - The function returns X and y as numpy arrays.
    - The number of rows in X matches the number of rows in the input data.
    - The number of rows in y matches the number of rows in the input data.
    """
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    assert isinstance(X, np.ndarray), "X should be a numpy array"
    assert isinstance(y, np.ndarray), "y should be a numpy array"
    assert X.shape[0] == sample_data.shape[0], "X should have the same number of rows as the input data"
    assert len(y) == sample_data.shape[0], "y should have the same number of rows as the input data"

# TODO: implement the second test. Change the function name and input as needed
def test_train_model(sample_data):
    """
    Tests the train_model function to ensure it trains a RandomForestClassifier.

    This test verifies that:
    - The function returns a trained model.
    - The returned model is an instance of RandomForestClassifier.
    """
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier), "Model should be a RandomForestClassifier"

# TODO: implement the third test. Change the function name and input as needed
def test_inference(sample_data):
    """
    Tests the inference function to ensure it generates predictions correctly.

    This test verifies that:
    - The function returns predictions as a numpy array.
    - The number of predictions matches the number of rows in the input data.
    """
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray), "Predictions should be a numpy array"
    assert len(preds) == X.shape[0], "Number of predictions should match the number of rows in X"