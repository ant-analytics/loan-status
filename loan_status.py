# region ========= Import Libraries =========
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC

import pandas as pd
import seaborn as sns
from src.data_cleaning import *
from src.EDA import *
# endregion ========= Import Libraries =========

# region ========= Load Data =========
raw_data = pd.read_csv('/mnt/d/Beta/data/raw_loan_data.csv')

# endregion ========= Load Data =========

# region ========= EDA =========
# Make a dashboard
# import matplotlib.pyplot as plt
# numerric_featues = raw_data.select_dtypes(exclude='object').columns.tolist()
# categorical_features = raw_data.select_dtypes(include='object').columns.tolist()
# all_features = numerric_featues + categorical_features

# mosaic = """
# ABCDE
# FGHIJ
# KLMNO
# """
# fig, axs = plt.subplot_mosaic(mosaic, figsize=(15, 10), layout='constrained')

# for i, (ax, feature) in enumerate(zip(axs.values(), all_features)):
#     if feature in numerric_featues:
#         sns.histplot(raw_data[feature], ax=ax)
#         ax.set_title(f'{feature} Distribution')
#         ax.set_xlabel(feature)
#         ax.set_ylabel('Frequency')
#     else:
#         sns.countplot(y=feature, data=raw_data, ax=ax)
#         ax.set_title(f'Count plot of {feature}')
#         ax.set_xlabel(feature)
#         ax.set_ylabel('Count')

# # delete unused subplots
# for j in range(i+1, len(axs)):
#     fig.delaxes(axs[j])

# # adjust layout
# plt.tight_layout()
# plt.show()

# endregion ========= EDA =========

# region ====== Data Splitting ======
def split_data(data, target, test_size=0.25, val_size=0.25, random_state=42):
    X = data.drop(target, axis=1)
    y = data[target]
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=random_state, stratify=y)
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size / (test_size + val_size), random_state=random_state, stratify=y_temp)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = split_data(raw_data, 'loan_status')

# X_train, X_temp, y_train, y_temp = train_test_split(
#     raw_data.drop('loan_status', axis=1),
#     raw_data['loan_status'],
#     test_size=0.5,
#     random_state=42,
#     stratify=raw_data['loan_status'])

# X_val, X_test, y_val, y_test = train_test_split(
#     X_temp, y_temp,
#     test_size=0.5,
#     random_state=42,
#     stratify=y_temp)
   

# endregion ====== Data Splitting ======

# region ====== Data Preprocessing ======
# Stratified sampling

# Data cleaning and transformation
# Detect and remove outliers

# Remove outliers from the training data
# X_train, y_train, outliers = remove_outliers(X_train, y_train, numerric_featues)

# Column transformation
col_transformer = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerric_featues),
        ('cat', OrdinalEncoder(), categorical_features)], remainder='passthrough')

#  Logistic Regression Pipeline
logit_pipe = Pipeline(steps=[
    ('preprocessor', col_transformer),
    ('classifier', LogisticRegression())])

# Train Model
# No need to assign the model to a variable as we can access it from the pipeline
logit_pipe.fit(X_train, y_train)

# Predict
y_pred = logit_pipe.predict(X_test)

# region ====== Model Evaluation ======
print(f"Model coefficients: {logit_pipe.named_steps['classifier'].coef_}")
print(f"Model intercept: {logit_pipe.named_steps['classifier'].intercept_}")
print(f"Classes: {logit_pipe.named_steps['classifier'].classes_}")
print(f"Number of iterations: {logit_pipe.named_steps['classifier'].n_iter_}")

# Evaluate the model

print(f"Accuracy: {logit_pipe.score(X_test, y_test):.2f}")
print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification report:\n{classification_report(y_test, y_pred)}")

# Plot ROC curve
RocCurveDisplay.from_estimator(logit_pipe, X_test, y_test)
plt.title('ROC Curve for Logistic Regression Model')
plt.show()

# Plot feature importance
# importance = pipe.named_steps['classifier'].coef_[0]
# feature_names = numerric_featues + categorical_features
# feature_importance = pd.Series(importance, index=feature_names).sort_values(ascending=False)

# sns.barplot(x=feature_importance, y=feature_importance.index)
# plt.title('Feature Importance')
# plt.show()
# endregion

# region ========= Model SVC =========
# this uses the Suport Vector Classification model
from sklearn.svm import SVC

# Create a pipeline
svc_pipe = Pipeline(steps=[
    ('preprocessor', col_transformer),
    ('classifier', SVC())])

svc_pipe.fit(X_train, y_train)
y_svc_pred = svc_pipe.predict(X_test)

# Print key information from the SVC model
print(f"Accuracy_SVC: {svc_pipe.score(X_test, y_test):.2f}")
print(f"Confusion matrix_SVC:\n{confusion_matrix(y_test, y_svc_pred)}")
print(f"Classification report_SVC:\n{classification_report(y_test, y_svc_pred)}")

# Plot ROC curve
RocCurveDisplay.from_estimator(svc_pipe, X_test, y_test)
plt.title('ROC Curve for SVC Model')
plt.show()

# Make confusion matrix plot
cm_svc = confusion_matrix(y_test, y_svc_pred)
sns.heatmap(cm_svc, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for SVC Model')
# endregion ========= Model Tuning =========

# region ========= Model Evaluation =========

# endregion ========= Model Evaluation =========
