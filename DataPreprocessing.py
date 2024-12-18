
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def DataPreprocessing(inputData):
    """
    Data preprocessing for the entire DataFrame.

    Parameters:
        inputData (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    # Create a copy of the data
    data = inputData.copy()

    # Check for anomalies: replace values >100% with NaN
    columns_to_check = ['fg', 'x3p', 'ft']
    for col in columns_to_check:
        if col in data.columns:
            data[col] = data[col].apply(lambda x: np.nan if pd.notna(x) and x > 100 else x)

    # Split into groups
    max_pts = data['pts'].max(skipna=True)
    prah_rozdeleni = np.round(max_pts / 2)

    index_sportovce_nad = data['pts'] > prah_rozdeleni
    index_sportovce_pod = ~index_sportovce_nad

    group_nad = data.loc[index_sportovce_nad].copy()
    group_pod = data.loc[index_sportovce_pod].copy()

    # Fill NaN with median values within groups
    numeric_columns = group_nad.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        group_nad[column] = group_nad[column].fillna(group_nad[column].median())
        group_pod[column] = group_pod[column].fillna(group_pod[column].median())

    # Combine the data back
    data.loc[group_nad.index, numeric_columns] = group_nad[numeric_columns]
    data.loc[group_pod.index, numeric_columns] = group_pod[numeric_columns]

    # Example of correlation calculation (if needed)
    correlation_matrix = data.select_dtypes(include=[np.number]).corr()
    correlation_with_pts = correlation_matrix['pts']

    return data
