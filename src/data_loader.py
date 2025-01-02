import pandas as pd
def load_data(raw_data_path, metadata_path):
    raw_data = pd.read_csv(raw_data_path)
    metadata = pd.read_csv(metadata_path, sep='\t', header=None, names=['Feature', 'Description', 'Type'])
    num_features = raw_data.select_dtypes(exclude='object').columns.tolist()
    cat_features = raw_data.select_dtypes(include='object').columns.tolist()
    return raw_data, num_features, cat_features, metadata