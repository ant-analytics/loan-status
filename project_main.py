import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_cleaning import *
from src.data_eda import *
# endregion ========= Import Libraries =========

# region ========= Load Data =========
raw_data = pd.read_csv('./data/raw_loan_data.csv')
metadata_path = './data/metadata.txt'
metadata = pd.read_csv(metadata_path, sep='\t', header=None, names=['Feature', 'Description', 'Type'])
num_features = raw_data.select_dtypes(exclude='object').columns.tolist()
cat_features = raw_data.select_dtypes(include='object').columns.tolist()
all_features = num_features + cat_features
# endregion ========= Load Metadata =========