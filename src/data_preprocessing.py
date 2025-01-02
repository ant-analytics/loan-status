from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


def split_data(raw_data):
    X = raw_data.drop(['loan_status'], axis=1)
    y = raw_data['loan_status']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_data(X_train, X_val, X_test, num_features, cat_features):
    col_transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), [feature for feature in num_features if feature != 'loan_status']),
            ('cat', OrdinalEncoder(), cat_features)], remainder='passthrough')
    X_train_transform = col_transformer.fit_transform(X_train)
    X_val_transform = col_transformer.transform(X_val)
    X_test_transform = col_transformer.transform(X_test)
    return X_train_transform, X_val_transform, X_test_transform


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

def split_data(data, target, stratify_feature=None, test_size=0.25, val_size=0.25, random_state=42):
    """Docstring for split_data
    Split dataset into training, validation, and test sets    
    :param data: The dataset to split
    :param target: The target column
    :param stratify_feature: The feature to use for stratification, defaults to None
    :param test_size: The test size, defaults to 0.25
    :param val_size: The validation size, defaults to 0.25
    :param random_state: The random state, defaults to 42
    ...   """
    
    X = data.drop(target, axis=1)
    y = data[target]
    
    stratify_param = data[stratify_feature] if stratify_feature else None
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=random_state, stratify=stratify_param)
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size / (test_size + val_size), random_state=random_state, stratify=stratify_param.loc[y_temp.index] if stratify_feature else None)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Data spliting use hash
import hashlib

def hash_function(value):
    """Generate a hash value for a given input."""
    return int(hashlib.md5(str(value).encode('utf-8')).hexdigest(), 16)

def assign_subset(hash_value, train_ratio=0.6, val_ratio=0.2):
    """
    Assign a data point to a subset based on its hash value.

    Parameters:
    hash_value (int): The hash value of the data point.
    train_ratio (float, optional): The ratio of data points to be assigned to the training set. Default is 0.6.
    val_ratio (float, optional): The ratio of data points to be assigned to the validation set. Default is 0.2.

    Returns:
    str: The subset to which the data point is assigned. It can be 'train', 'val', or 'test'.

    Example:
    >>> assign_subset(42)
    'train'
    >>> assign_subset(85)
    'test'
    >>> assign_subset(65, train_ratio=0.5, val_ratio=0.3)
    'val'
    """
    """Assign a data point to a subset based on its hash value."""
    if hash_value % 100 < train_ratio * 100:
        return 'train'
    elif hash_value % 100 < (train_ratio + val_ratio) * 100:
        return 'val'
    else:
        return 'test'

def stratify_samples(df, id_column, train_ratio=0.6, val_ratio=0.2):
    """
    Stratify samples based on hash values of an identifier column.
    This function divides the input DataFrame into training, validation, and test sets
    based on the hash values of a specified identifier column. The hash values are used
    to ensure that the split is consistent and reproducible.
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data to be stratified.
    id_column (str): The name of the column in the DataFrame to be used as the identifier for hashing.
    train_ratio (float, optional): The proportion of the data to be used for the training set. Default is 0.6.
    val_ratio (float, optional): The proportion of the data to be used for the validation set. Default is 0.2.
    Returns:
    tuple: A tuple containing three DataFrames: (train_data, val_data, test_data).
        - train_data (pandas.DataFrame): The training set.
        - val_data (pandas.DataFrame): The validation set.
        - test_data (pandas.DataFrame): The test set.
    Note:
    The remaining proportion (1 - train_ratio - val_ratio) will be used for the test set.
    Example:
    >>> train_data, val_data, test_data = stratify_samples(df, 'id', train_ratio=0.7, val_ratio=0.15)
    """
    df['hash'] = df[id_column].apply(hash_function)
    df['subset'] = df['hash'].apply(assign_subset, train_ratio=train_ratio, val_ratio=val_ratio)
    
    train_data = df[df['subset'] == 'train'].drop(['hash', 'subset'], axis=1)
    val_data = df[df['subset'] == 'val'].drop(['hash', 'subset'], axis=1)
    test_data = df[df['subset'] == 'test'].drop(['hash', 'subset'], axis=1)
    
    return train_data, val_data, test_data

def update_stratified_samples(df, id_column, existing_subsets, train_ratio=0.6, val_ratio=0.2):
        """
        Update stratified samples to ensure old members remain in the same subset when new elements are added.

        Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data to be stratified.
        id_column (str): The name of the column in the DataFrame to be used as the identifier for hashing.
        existing_subsets (pandas.Series): A Series containing the existing subset assignments.
        train_ratio (float, optional): The proportion of the data to be used for the training set. Default is 0.6.
        val_ratio (float, optional): The proportion of the data to be used for the validation set. Default is 0.2.

        Returns:
        tuple: A tuple containing three DataFrames: (train_data, val_data, test_data).
            - train_data (pandas.DataFrame): The training set.
            - val_data (pandas.DataFrame): The validation set.
            - test_data (pandas.DataFrame): The test set.
        
        Example:
            >>> import pandas as pd
            >>> data = {'id': [1, 2, 3, 4, 5, 6], 'feature': [10, 20, 30, 40, 50, 60]}
            >>> df = pd.DataFrame(data)
            >>> existing_subsets = pd.Series(['train', 'val', 'test', 'train', 'val', None])
            >>> train_data, val_data, test_data = update_stratified_samples(df, 'id', existing_subsets)
            >>> print(train_data)
               id  feature
            0   1       10
            3   4       40
            >>> print(val_data)
               id  feature
            1   2       20
            4   5       50
            >>> print(test_data)
               id  feature
            2   3       30
            5   6       60
        """
        # Assign existing subsets
        df['subset'] = existing_subsets

        # Identify new elements
        new_elements = df['subset'].isnull()

        # Assign new elements to subsets based on hash values
        df.loc[new_elements, 'hash'] = df.loc[new_elements, id_column].apply(hash_function)
        df.loc[new_elements, 'subset'] = df.loc[new_elements, 'hash'].apply(assign_subset, train_ratio=train_ratio, val_ratio=val_ratio)

        # Split the data into train, val, and test sets
        train_data = df[df['subset'] == 'train'].drop(['hash', 'subset'], axis=1)
        val_data = df[df['subset'] == 'val'].drop(['hash', 'subset'], axis=1)
        test_data = df[df['subset'] == 'test'].drop(['hash', 'subset'], axis=1)

        return train_data, val_data, test_data