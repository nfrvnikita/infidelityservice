"""Py file for encoding features"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def encode_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the input dataset using OneHotEncoder

    Args:
        df (pd.DataFrame): Input data

    Returns:
        pd.DataFrame: Encoded dataset
    """
    encoder = OneHotEncoder()
    encoded_df = pd.DataFrame(encoder.fit_transform(df[['age', 'yrs_married', 'occupation',
                                                        'occupation_husb']]).toarray(),
                              columns=encoder.get_feature_names_out(['age', 'yrs_married',
                                                                     'occupation', 'occupation_husb']))
    df = pd.concat([df.drop(columns=['age', 'yrs_married', 'occupation', 'occupation_husb']), encoded_df], axis=1)
    return df
