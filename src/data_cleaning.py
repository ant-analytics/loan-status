import pandas as pd

def load_and_remove_nan(file_path):
    """"Docstring for load_and_remove_nan
    
    :param file_path: File in csv format
    :type file_path: Path to file 
    return dataframe with no NaN value
    """
    print(f"Start loading data and remove NaN values")
    print(f"Start loading data")
    raw_data = pd.read_csv(file_path)
    print(f"Finish loading data")
    print(f"Start removing NaN values")
    print(f"Detect NaN values ...")
    for col in raw_data.columns:
        print(f"Column: {col} has {raw_data[col].isnull().sum()} NaN values --> Remove {raw_data[col].isnull().sum()} sample(s)")
    print(f"Start remove NaN values")
    print(f"Finish remove NaN values")
    analysis_data = raw_data.dropna()
    return raw_data, analysis_data
