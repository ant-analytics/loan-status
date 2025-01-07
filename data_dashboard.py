
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

# region ========= EDA =========
# Make a dashboard
import matplotlib.pyplot as plt
import matplotlib.style
matplotlib.style.use('seaborn-v0_8-whitegrid') # Set global theme style

# Calculate descriptive statistics
desc_stats = raw_data.describe().T[['min', 'mean', 'std', 'max']].sort_values(by='max', ascending=False).applymap(lambda x: f'{x:,.2f}')

# Create a mosaic plot layout dynamically from all features
num_rows = (len(all_features) + 2) // 3  # Calculate number of rows needed
mosaic = []
for i in range(num_rows):
    row = all_features[i*3:(i+1)*3]
    while len(row) < 3:
        row.append('reserve')  # Fill the row with 'reserve' if less than 3 features
    mosaic.append(row)

mosaic_final = [
    ['info', 'person_gender', 'person_education', 'person_home_ownership'],
    ['person_age', 'person_income', 'loan_amnt', 'credit_score'],
    ['person_emp_exp','loan_percent_income', 'loan_int_rate', 'loan_status'],
    ['cb_person_cred_hist_length','loan_intent', 'previous_loan_defaults_on_file', 'reserve_1']
    ]

fig, axs = plt.subplot_mosaic(mosaic_final, figsize=(24, 18), layout='constrained')
fig.patch.set_facecolor('#f0f0f0')  # Set background color for the figure

def plot_features(data, num_features, cat_features, axs, metadata):
    """"Docstring for plot_features
    
    :param data: The dataset containing the features
    :type data: pd.DataFrame
    :param num_features: List of numerical features
    :type num_features: list
    :param cat_features: List of categorical features
    :type cat_features: list
    :param axs: Axes for plotting
    :type axs: dict
    :param metadata: Metadata containing feature descriptions
    :type metadata: pd.DataFrame
    """
    # Plot loan_status as count bar plot
    loan_status_counts = data['loan_status'].value_counts().sort_values(ascending=False)
    axs['loan_status'].bar(loan_status_counts.index, loan_status_counts.values, color='#4CAF50')
    axs['loan_status'].set_title('Bar Plot of Loan Status', fontweight='bold', fontsize=14)
    axs['loan_status'].set_xlabel('Loan Status', fontsize=12)
    axs['loan_status'].set_ylabel('Count', fontsize=12)
    axs['loan_status'].set_xticks([0, 1])  # Set x-axis ticks to only 0 and 1
    axs['loan_status'].set_xticklabels(['Approve', 'Reject'], fontsize=10)  # Set tick labels to Approve and Reject
    # Add count numbers in the middle of the bars with thousand separator
    for i, count in enumerate(loan_status_counts.values):
        axs['loan_status'].text(i, count / 2, f'{count:,}', ha='center', va='center', color='black', fontsize=14)
    
    # Plot histogram for credit_score
    axs['credit_score'].hist(data['credit_score'], bins=30, color='#2196F3', edgecolor='black')
    axs['credit_score'].set_title('Histogram of Credit Score', fontweight='bold', fontsize=14)
    axs['credit_score'].set_xlabel('Credit Score', fontsize=12)
    axs['credit_score'].set_ylabel('Frequency', fontsize=12)
    
    
    for feature in cat_features:
        counts = data[feature].value_counts().sort_values(ascending=False)
        description = metadata.loc[metadata['Feature'] == feature, 'Description'].values[0]
        axs[feature].bar(counts.index, counts.values, color='#FF5722')
        axs[feature].set_title(f'Bar Plot of {description}', fontweight='bold', fontsize=14)
        axs[feature].set_ylabel('Count', fontsize=12)
        axs[feature].set_xlabel('', fontsize=12)  # Hide x-axis label

    for feature in num_features:
        if feature not in ['loan_status', 'credit_score']:
            description = metadata.loc[metadata['Feature'] == feature, 'Description'].values[0]
            axs[feature].boxplot(data[feature], vert=False, patch_artist=True, boxprops=dict(facecolor='#FFC107'))
            axs[feature].set_title(f'Box Plot of {description}', fontweight='bold', fontsize=14)
            axs[feature].set_xlabel('', fontsize=12)  # Hide x-axis label
            axs[feature].yaxis.set_visible(False)  # Hide y-axis
    # Set x-axis labels for 'loan_intent' to be more readable
    axs['loan_intent'].set_xticklabels(axs['loan_intent'].get_xticklabels(), rotation=45, ha='right', fontsize=10)

    # Add conclusion text
    conclusion_text = (
        f"Exploratory Data Analysis Insights:\n"
        f"1. Presence of outliers in 'Age' and 'Years of Experience'.\n"
        f"2. Significant scale differences in 'Income' and 'Age'.\n"
        f"3. Imbalance observed in 'Loan Status'.\n"
        f"4. 'Credit Score' exhibits outliers.\n"
        f"5. There is no missing values in the dataset."
    )
    axs['reserve_1'].text(0.01, 0.99, conclusion_text, ha='left', va='top', fontsize=14, wrap=True, linespacing=2)
    axs['reserve_1'].set_axis_off()  # Hide the axis
    
    # Add info text
    info_text = (
        f"**Dataset Information:**\n"
        f"**Number of Samples:** {data.shape[0]:,}\n"
        f"**Number of Features:** {data.shape[1]:,}\n"
        f"**Numerical Features:** {len(num_features):,}\n"
        f"**Categorical Features:** {len(cat_features):,}\n"
    )
    axs['info'].text(0.01, 0.99, info_text, ha='left', va='top', fontsize=14, wrap=True, linespacing=2)
    axs['info'].set_axis_off()  # Hide the axis

plot_features(raw_data, num_features, cat_features, axs, metadata)

# Save the plot
plt.savefig('/mnt/d/Beta/plots/eda_dashboard.png', dpi=384, bbox_inches='tight')
plt.show()
# endregion ========= EDA =========