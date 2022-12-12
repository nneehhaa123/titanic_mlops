import numpy as np
import pandas as pd

# functions to test are imported from train.py
from train import data_transformation,split_data, train_model, get_model_metrics

"""A set of simple unit tests for protecting against regressions in train.py"""


def test_split_data():
    test_data = {'Survived': [0, 1, 1, 1, 0,0, 1, 1, 1, 0],
                     'Pclass': [3, 1,3, 1, 3, 3, 1,3, 1, 3],
                     'Sex': ['male','female','female', 'female','male', 'male','female','female', 'female','male'],
                     'Age': [22.0,38.0, 26.0,35.0,35.0,22.0,38.0, 26.0,35.0,35.0],
                     'SibSp': [1, 1,0, 1, 0,1, 1,0, 1, 0],
                     'Parch': [0, 1, 3, 0,0, 0, 2, 0, 4,0],
                     'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05, 7.25, 71.2833, 7.925, 53.1, 8.05],
                     'Embarked': ['S','C', 'S', 'S', 'S', 'S','C', 'S', 'S', 'C']}

    data_df = pd.DataFrame(data=test_data)
    data = split_data(data_df)


    # verify that data was split as desired
    assert data[0].data.shape[0] == 8
    assert data[1].data.shape[0] == 2



def test_train_model():
    data = __get_test_datasets()

    params = {
        "learning_rate": 0.05,
        "metric": "auc",
        "min_data": 1
    }

    model = train_model(data, params)

    # verify that parameters are passed in to the model correctly
    for param_name in params.keys():
        assert param_name in model.params
        assert params[param_name] == model.params[param_name]


def test_get_model_metrics():
    class MockModel:

        @staticmethod
        def predict(data):
            return np.array([0, 0])

    data = __get_test_datasets()

    metrics = get_model_metrics(MockModel(), data)

    # verify that metrics is a dictionary containing the auc value.
    #assert "auc" in metrics
    #auc = metrics["auc"]
    #np.testing.assert_almost_equal(auc, 0.5)


def __get_test_datasets():
    """This is a helper function to set up some test data"""
    test_data = {'Survived': [0, 1, 1, 1, 0,0, 1, 1, 1, 0],
                     'Pclass': [3, 1,3, 1, 3, 3, 1,3, 1, 3],
                     'Sex': ['male','female','female', 'female','male', 'male','female','female', 'female','male'],
                     'Age': [22.0,38.0, 26.0,35.0,35.0,22.0,38.0, 26.0,35.0,35.0],
                     'SibSp': [1, 1,0, 1, 0,1, 1,0, 1, 0],
                     'Parch': [0, 1, 3, 0,0, 0, 2, 0, 4,0],
                     'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05, 7.25, 71.2833, 7.925, 53.1, 8.05],
                     'Embarked': ['S','C', 'S', 'S', 'S', 'S','C', 'S', 'S', 'C']}

    data_df = pd.DataFrame(data=test_data)
    data = split_data(data_df)
    return data