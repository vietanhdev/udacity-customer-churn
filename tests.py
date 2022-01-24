"""Test for Churn Predictor library
   Author: Viet-Anh Nguyen
   Created at: 24/01/2022
"""

import logging
import os
import pathlib
import shutil

from churn_library import CustomerChurnPredictor

pathlib.Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    filename='./logs/test_log.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    """Test data import
    """
    try:
        data_path = "./data/bank_data.csv"
        predictor = CustomerChurnPredictor(
            data_path, log_path='./logs/churn_library.log')
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert predictor.data.shape[0] > 0
        assert predictor.data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    """Test perform eda function
    """
    try:
        shutil.rmtree("eda", ignore_errors=True)
        data_path = "./data/bank_data.csv"
        predictor = CustomerChurnPredictor(
            data_path, log_path='./logs/churn_library.log')
        predictor.perform_eda(output_folder="eda")
        assert os.path.isfile("eda/churn_histogram.png")
        assert os.path.isfile("eda/correlation.png")
        assert os.path.isfile("eda/customer_age_histogram.png")
        assert os.path.isfile("eda/marital_status.png")
        assert os.path.isfile("eda/total_trans_ct.png")
        logging.info("Testing test_eda: SUCCESS")
    except Exception as err:
        logging.error("Test not passed: {err}")
        raise err


def test_prepare_data():
    """Test data preparation
    """
    try:
        shutil.rmtree("eda", ignore_errors=True)
        data_path = "./data/bank_data.csv"
        predictor = CustomerChurnPredictor(
            data_path, log_path='./logs/churn_library.log')
        predictor.perform_eda(output_folder="eda")
        predictor.prepare_data(test_size=0.3)

        for cat in CustomerChurnPredictor.cat_columns:
            if "{cat}_Churn" not in predictor.data:
                err = 'Error in encode_categorical_features. Missing: {cat}'
                logging.error(err)
                raise Exception(err)

        if predictor.x_train is None or predictor.x_test is None \
                or predictor.y_train is None or predictor.y_test is None:
            err = 'Error in data splitting'
            logging.error(err)
            raise Exception(err)

        logging.info("Testing test_prepare_data: SUCCESS")
    except Exception as err:
        logging.error("Test not passed: {err}")
        raise err


def test_train_models():
    """Test train_models
    """
    try:
        shutil.rmtree("images", ignore_errors=True)
        shutil.rmtree("models", ignore_errors=True)
        data_path = "./data/bank_data.csv"
        predictor = CustomerChurnPredictor(
            data_path, log_path='./logs/churn_library.log')
        predictor.perform_eda(output_folder="eda")
        predictor.prepare_data(test_size=0.3)
        predictor.train_models(output_folder="models")

        assert os.path.isfile("models/rfc_model.pkl")
        assert os.path.isfile("models/logistic_model.pkl")

        logging.info("Testing test_train_models: SUCCESS")
    except Exception as err:
        logging.error("Test not passed: {err}")
        raise err


def test_evaluation():
    """Test model evaluation
    """
    try:
        shutil.rmtree("images", ignore_errors=True)
        shutil.rmtree("models", ignore_errors=True)
        shutil.rmtree("results", ignore_errors=True)
        data_path = "./data/bank_data.csv"
        predictor = CustomerChurnPredictor(
            data_path, log_path='./logs/churn_library.log')
        predictor.perform_eda(output_folder="eda")
        predictor.prepare_data(test_size=0.3)
        predictor.train_models(output_folder="models")
        predictor.evaluation(output_folder="results")

        assert os.path.isfile("results/lrc_roc_curve.png")
        assert os.path.isfile("results/cv_rfc_roc_curve.png")

        logging.info("Testing test_evaluation: SUCCESS")
    except Exception as err:
        logging.error("Test not passed: {err}")
        raise err


def test_feature_importance_plot():
    """Test feature_importance_plot
    """
    try:
        shutil.rmtree("images", ignore_errors=True)
        shutil.rmtree("models", ignore_errors=True)
        shutil.rmtree("results", ignore_errors=True)
        pathlib.Path("results").mkdir(exist_ok=True)
        data_path = "./data/bank_data.csv"
        predictor = CustomerChurnPredictor(
            data_path, log_path='./logs/churn_library.log')
        predictor.perform_eda(output_folder="eda")
        predictor.prepare_data(test_size=0.3)
        predictor.train_models(output_folder="models")
        predictor.feature_importance_plot("results/feature_importance.png")
        assert os.path.isfile("results/feature_importance.png")

        logging.info("Testing feature_importance_plot: SUCCESS")
    except Exception as err:
        logging.error("Test not passed: {err}")
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_prepare_data()
    test_train_models()
    test_evaluation()
    test_feature_importance_plot()
