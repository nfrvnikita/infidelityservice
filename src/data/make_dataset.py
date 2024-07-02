"""Function for obtaining train and test samples after all transformations"""
import os
import typing as tp
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from src.data.encode_data import encode_dataset

load_dotenv()
RAW_DATA_PATH = os.getenv('RAW_DATA_PATH')


def make_dataset(input_data: str) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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


def tensors_data() -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts dataset into PyTorch tensors for training and validation.

    Returns:
        tp.Tuple[torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor]: Tensors for training and validation.
    """
    X_train, X_valid, y_train, y_valid = make_dataset(RAW_DATA_PATH)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_valid_tensor = torch.tensor(X_valid.values, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid.values, dtype=torch.float32)

    return X_train_tensor, X_valid_tensor, y_train_tensor, y_valid_tensor
