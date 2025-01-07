import pandas as pd

def load_data(raw_data_path, metadata_path):
    """
    Load raw data and metadata from specified file paths.

    Parameters:
    raw_data_path (str): The file path to the raw data CSV file.
    metadata_path (str): The file path to the metadata CSV file.

    Returns:
    tuple: A tuple containing:
        - raw_data (pd.DataFrame): The raw data loaded from the CSV file.
        - num_features (list): A list of numerical feature names.
        - cat_features (list): A list of categorical feature names.
        - metadata (pd.DataFrame): The metadata loaded from the CSV file.
    """
    raw_data = pd.read_csv(raw_data_path)
    metadata = pd.read_csv(metadata_path, sep='\t', header=None, names=['Feature', 'Description', 'Type'])
    num_features = raw_data.select_dtypes(exclude='object').columns.tolist()
    cat_features = raw_data.select_dtypes(include='object').columns.tolist()
    
    return raw_data, num_features, cat_features, metadata