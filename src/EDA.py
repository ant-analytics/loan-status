import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

def describe_data(df):
    """ 
    
    Docstring for describe_data

    :param df: input dateframe
    :type df: dataframe
    an adavanced version of pd.describe function """
    print(f"Key information about dataset")
    advance_describe = df.describe(include='all')
    advance_describe.loc['dtype'] = df.dtypes
    advance_describe.loc['NaN count'] = df.isnull().sum()
    return advance_describe

def remove_outliers(df):
    """
    Function to remove outliers in multiple columns in dataframe (df)
    
    Parameters:
    - df: pandas DataFrame containing the data.
    
    Returns:
    - pandas DataFrame with outliers removed.
    """
    for col in df.select_dtypes(include='number').columns.to_list():
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df  = df[(df[col] >=lower_bound) & (df[col] <= upper_bound)]
        return df
    
def boxplot_dataframe(df, ncols=3):
    """Docstring for box_plot_grid
    
    Plot boxplot for dataframe to detect outliers.
    
    Parameters:
    - df: pandas DataFrame containing the data.
    - ncols: Number of columns in the plot grid (default is 3)
    """
    # filter only numeric columns
    numeric_df = df.select_dtypes(include='number')

    # get nrow and ncol for grid
    ncols = ncols
    nrows = (len(numeric_df.columns) + ncols - 1) // ncols # Automatic calculate number of rows

    # create subpplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5))

    # Flatten axs for looping
    axes = axes.flatten()

    for i, col in enumerate(numeric_df.columns):
        sns.boxplot(x=numeric_df[col], ax=axes[i])
        axes[i].set_title(f"Box plot of {col}")

    # Remove any unused subplot
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_histogram(df, ncols=3):
    """
    Plot histogram for all columns in dataframe with adjustment to datatype.
    
    Parameters:
    - df: pandas DataFrame containing the data.
    - ncols: Number of columns in the plot grid (default is 3).
    """
    # Separate numeric and object columns
    numeric_columns = df.select_dtypes(exclude='object').columns.tolist()
    object_columns = df.select_dtypes(include='object').columns.tolist()

    # Combine numeric and object columns, with numeric columns first
    all_columns = numeric_columns + object_columns

    # Calculate number of rows for grid
    nrows = (len(all_columns) + ncols - 1) // ncols

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5))

    # Flatten axes for looping
    axes = axes.flatten()

    # Iterate over all columns in the desired order
    for i, col in enumerate(all_columns):
        ax = axes[i]
        if col in object_columns:
            #  Sort the categories by frequency
            sorted_categories = df[col].value_counts().index
            sns.countplot(x=col, data=df, ax=ax, order=sorted_categories)
            ax.set_title(f"Count plot of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
        else:
            sns.histplot(df[col], ax=ax, kde=True, bins=20, color='skyblue', line_kws={'color': 'red'})
            ax.set_title(f"Histogram of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')

    # Remove any unused subplot
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    plt.show()

def scatter_plot(df, target_feature, ncols=3, jitter_width=0.3):
    """
    Scatter plot all features against a specific target feature.

    :param df: Input dataframe
    :type df: pd.DataFrame
    :param target_feature: Feature to plot against
    :type target_feature: str
    :param ncols: Number of columns in the subplot grid, default is 3
    :type ncols: int
    """
    if target_feature not in df.columns:
        raise ValueError(f"{target_feature} is not a column in the DataFrame.")

    # Separate target from other features
    features = df.drop(columns=[target_feature]).columns
    nrows = (len(features) + ncols - 1) // ncols  # Calculate the number of rows

    # Create subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    axes = axes.flatten()

    # Plot each feature
    for i, feature in enumerate(features):
        ax = axes[i]
        if df[feature].dtype in ['object', 'category']:
            sns.stripplot(x=feature, y=target_feature, data=df, ax=ax, jitter=jitter_width, palette='Set2', alpha=0.7)
            ax.set_title(f"{feature} vs {target_feature}")
        else:
            sns.scatterplot(x=feature, y=target_feature, data=df, ax=ax, color='blue', alpha=0.7)
            ax.set_title(f"{feature} vs {target_feature}")

        ax.set_xlabel(feature)
        ax.set_ylabel(target_feature)

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    plt.show()

def one_hot_encoder(df, columns_to_encode):
    """ 
    
    Docstring for one_hot_encoder: take a dataframe and list of columns to encode return the modified dataframe.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns_to_encode (list): List of column names to one-hot encode.
    
    Returns:
        pd.DataFrame: A new DataFrame with one-hot encoded columns. """
    try:
        # Initialise OneHotEncoder
        one_hot_encoder = OneHotEncoder(sparse_output= False, drop=None) # set sparse_output = false to get array output
        
        # Fit and transform the data
        encoded_array = one_hot_encoder.fit_transform(df.loc[:, columns_to_encode])

        # Get names of one-hot-encode columns 
        encoded_column_names = one_hot_encoder.get_feature_names_out(columns_to_encode)

        # Create a dataframe for one-hot-encoded columns
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_column_names, index=df.index)

        # Concatenate with original dataframe and exlcude the column to be encoded
        df = pd.concat([df.drop(columns=columns_to_encode), encoded_df], axis=1)
    
    except KeyError as e:
        print(f"Error: {e}. One or more columns to encode do not exist in the DataFrame.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return df

def ordinal_encode_column_inplace(df, columns_to_encode):
    """
    Docstring for one_hot_encoder: take a dataframe and list of columns to encode return the modified dataframe.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns_to_encode (list): List of column names to one-hot encode.
    
    Returns:
        pd.DataFrame: A modified DataFrame with one-hot encoded columns. """
    try:
        # Initialise the OrdinalEncoder
        ordinal_encoder = OrdinalEncoder()

        # Perform the encoding and replace the original column with the encoded one
        encoded_values = ordinal_encoder.fit_transform(df.loc[:, columns_to_encode])
        df[columns_to_encode] = encoded_values
    
    except KeyError as e:
        print(f"Error: {e}. One or more columns to encode do not exist in the DataFrame.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return df

def calculate_vif(df):
    """
    Calculates the Variance Inflation Factor (VIF) for each feature in the dataframe.

    Parameters:
    - df: pandas DataFrame containing the features for which VIF is to be calculated.

    Returns:
    - pandas DataFrame with features and their corresponding VIF values.
    """
    try:
        # Add constant (intercept) to the features
        X = add_constant(df)

        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data['Feature'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i).round(2) for i in range(X.shape[1])]

        # Round the VIF values to 2 decimal places
        # vif_data = vif_data.round({'VIF': 2})

    except Exception as e:
        print(f"An error occurred while calculating VIF: {e}")

    return vif_data