import os
import logging
import churn_library as cls
import pytest

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    """
	test data import - this example is completed for you to assist with the other test functions
	"""
    try:
        df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    pytest.df = df


def test_eda():
    """
	test perform eda function
	"""
    cls.perform_eda(pytest.df)
    directory = "./images/eda"
    files_names = ["dataset_description", "churn_dist", "customers_age", "martial_status_percentage",
                   "total_trans_ct", "features_correlations"]
    for file in files_names:
        try:
            assert os.path.exists(directory + "/" + file + ".png")
        except AssertionError as err:
            logging.error("Testing perform_eda: {} has not been created".format(file))
            raise err


def test_encoder_helper():
    """
	test encoder helper
	"""
    pass


def test_perform_feature_engineering():
    """
	test perform_feature_engineering
	"""
    pass


def test_train_models():
    """
	test train_models
	"""
    pass


if __name__ == "__main__":
    pass
