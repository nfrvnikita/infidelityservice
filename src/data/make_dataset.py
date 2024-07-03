"""Function for obtaining train and test samples after all transformations"""
import os
import typing as tp
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from src.data.encode_data import encode_dataset

load_dotenv()
RAW_DATA_PATH = os.getenv('RAW_DATA_PATH')


def make_dataset(input_data: str = RAW_DATA_PATH) -> tp.Tuple[pd.DataFrame,
                                                              pd.DataFrame,
                                                              pd.Series,
                                                              pd.Series]:
    """Function for obtaining traine and test samples after all transformations.

    Args:
        input_data (pd.DataFrame): Our raw data.

    Returns:
        tp.Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Train and test samples.
    """
    data = pd.read_csv(input_data)
    encoded_data = encode_dataset(data)

    X = encoded_data.drop(columns=['affairs'])
    y = encoded_data['affairs']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
