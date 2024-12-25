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

# Stratified sampling
from sklearn.model_selection import train_test_split

def stratified_sampling(df, stratify_columns, target_column, test_size=0.2, random_state=42):
    """
    Perform stratified sampling on the DataFrame based on multiple features.

    Parameters:
    - df: pandas DataFrame containing the data.
    - stratify_columns: List of column names to use for stratification.
    - target_column: The name of the target column.
    - test_size: The proportion of the dataset to include in the test split (default is 0.2).
    - random_state: Random seed for reproducibility (default is 42).

    Returns:
    - X_train, X_test, y_train, y_test: Stratified training and testing sets.
    """
    # Create a new column that combines the values of the stratify columns
    df['stratify_col'] = df[stratify_columns].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    # Perform stratified sampling using train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop([target_column, 'stratify_col'], axis=1), 
        df[target_column], 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df['stratify_col']
    )

    # Drop the temporary stratify column
    df.drop('stratify_col', axis=1, inplace=True)
    
    return X_train, X_test, y_train, y_test