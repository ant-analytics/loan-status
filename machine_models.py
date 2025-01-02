# region ========= Import Libraries =========
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay

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
# region ====== Data Splitting ======
# Set X and y
X = raw_data.drop(['loan_status'], axis=1)
y = raw_data['loan_status']

# Split the data into train, validation test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
# endregion ====== Data Splitting ======

# region ====== Data Preprocessing ======
# Data cleaning and transformation
# Column transformation
col_transformer = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [feature for feature in num_features if feature != 'loan_status']),
        ('cat', OrdinalEncoder(), cat_features)], remainder='passthrough')

# Get the column names
X_train_cols = X_train.columns.tolist()
X_train_transform = col_transformer.fit_transform(X_train)
X_val_transform = col_transformer.transform(X_val)
X_test_transform = col_transformer.transform(X_test)
# Outlier detection
# endregion ====== Data Preprocessing ======

# region  ====== Logistic Regression Model ======
logit_pipe = Pipeline(steps=[
    ('preprocessor', col_transformer),
    ('classifier', LogisticRegression())])

# Train Model
# No need to assign the model to a variable as we can access it from the pipeline
logit_pipe.fit(X_train, y_train)

# Predict
y_pred = logit_pipe.predict(X_val)

# Model Evaluation
print(f"Model coefficients: {logit_pipe.named_steps['classifier'].coef_}")
print(f"Model intercept: {logit_pipe.named_steps['classifier'].intercept_}")
print(f"Classes: {logit_pipe.named_steps['classifier'].classes_}")
print(f"Number of iterations: {logit_pipe.named_steps['classifier'].n_iter_}")

# Evaluate the model
print(f"Accuracy: {logit_pipe.score(X_val, y_val):.2f}")
print(f"Confusion matrix:\n{confusion_matrix(y_val, y_pred)}")
print(f"Classification report:\n{classification_report(y_val, y_pred)}")

# Plot ROC curve
RocCurveDisplay.from_estimator(logit_pipe, X_val, y_val)
plt.title('ROC Curve for Logistic Regression Model')
plt.show()
# endregion ====== Logistic Model ======

# region ========= Model SVC =========
# this uses the Suport Vector Classification model
from sklearn.svm import SVC

# Create a pipeline
svc_pipe = Pipeline(steps=[
    ('preprocessor', col_transformer),
    ('classifier', SVC())])

svc_pipe.fit(X_train, y_train)
y_svc_pred = svc_pipe.predict(X_val)

# Print key information from the SVC model
print(f"Accuracy_SVC: {svc_pipe.score(X_val, y_val):.2f}")
print(f"Confusion matrix_SVC:\n{confusion_matrix(y_val, y_svc_pred)}")
print(f"Classification report_SVC:\n{classification_report(y_val, y_svc_pred)}")

# Plot ROC curve
RocCurveDisplay.from_estimator(svc_pipe, X_val, y_val)
plt.title('ROC Curve for SVC Model')
plt.show()
# endregion ========= Model SVC ==========

# region ========= Random Forest Model =========

# endregion ========= Random Forest Model =========

# region ====== Model Evaluation =======
# Compare key information output from models
# Create a DataFrame to compare precision, recall, and other indicators
# Below is nested dictionary each row is a dictionary and column is also a dictionary
evaluation_metrics = {
    'Model': ['Logistic Regression', 'SVC'],
    'Accuracy': [logit_pipe.score(X_val, y_val), svc_pipe.score(X_val, y_val)],
    'Precision': [classification_report(y_val, y_pred, output_dict=True)['weighted avg']['precision'],
                  classification_report(y_val, y_svc_pred, output_dict=True)['weighted avg']['precision']],
    'Recall': [classification_report(y_val, y_pred, output_dict=True)['weighted avg']['recall'],
               classification_report(y_val, y_svc_pred, output_dict=True)['weighted avg']['recall']],
    'F1-Score': [classification_report(y_val, y_pred, output_dict=True)['weighted avg']['f1-score'],
                 classification_report(y_val, y_svc_pred, output_dict=True)['weighted avg']['f1-score']]
}

metrics_df = pd.DataFrame(evaluation_metrics).round(2)

# Plot the comparison of evaluation metrics
fig, axs_mod = plt.subplots(1, 2, figsize=(14, 6))
metrics_df.set_index('Model').plot(kind='bar', ax=axs_mod[0], width=0.7)  # Adjust the width to make bars closer
axs_mod[0].set_title('Comparison of Evaluation Metrics')
axs_mod[0].set_ylabel('Score')
axs_mod[0].set_xlabel('Model')
axs_mod[0].legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside the plot
axs_mod[0].tick_params(axis='x', rotation=0)

# Plot ROC curves for both models
RocCurveDisplay.from_estimator(logit_pipe, X_val, y_val, ax=axs_mod[1], name='Logistic Regression')
RocCurveDisplay.from_estimator(svc_pipe, X_val, y_val, ax=axs_mod[1], name='SVC')
axs_mod[1].set_title('ROC Curve Comparison')
plt.tight_layout()
plt.show()
# endregion ========= Model Evaluation =========

# region ========= Sequential Neural Network Model ==========
# keep track of the column transformation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import plot_model # for visualizing the model
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import datetime

# Create a TensorBoard Callback
# Set Up a Directory for Logs
log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# model checkpoint
checkpoint_callback = ModelCheckpoint(
    filepath=f'./model_save/model_checkpoint_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.weights.h5',
    save_weights_only=True,
    save_freq='epoch')

# Define the model
neural_model = keras.Sequential([
    Input(shape=(X_train.shape[1],), name='input_layer'),
    Dense(64, activation='relu', name='hidden_layer_1'),
    Dense(32, activation='relu', name='hidden_layer_2'),
    Dense(1, activation='sigmoid', name='output_layer')
])

# Compile the model
neural_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Visualize the model architecture
plot_model(neural_model, to_file='/mnt/d/Beta/model_architecture.png', show_shapes=True, show_layer_names=True)
neural_model_history = neural_model.fit(X_train_transform, y_train, epochs=100, batch_size=32, validation_data=(X_val_transform, y_val))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.gca().set_xlim(left=0)
plt.show()

# Evaluate the model

# make predictions

# endregion ========= Sequantial Neural Network Model =========

# region ========= Functional API Model Evaluation =========
#  model architecture
input_ = keras.layers.Input(shape=(X_train_transform.shape[1],))
hidden1 = keras.layers.Dense(64, activation='relu')(input_)
hidden2 = keras.layers.Dense(32, activation='relu')(hidden1)
hidden3 = keras.layers.Dense(16, activation='relu')(hidden2)
output_ = keras.layers.Dense(1, activation='sigmoid')(hidden3)
functional_model = keras.Model(inputs=input_, outputs=output_)

# Compile the model
functional_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'precision', 'recall'])
# fit the model
functional_history = functional_model.fit(X_train_transform, y_train, epochs=10, batch_size=32, validation_data=(X_val_transform, y_val), verbose=2, callbacks=[tensorboard_callback, checkpoint_callback])
# evaluate the model
# return the loss value & metrics values specified at compile for the model in test mode
model_evaluation = functional_model.evaluate(X_test_transform, y_test)
# make predictions
model_predict = functional_model.predict(X_test_transform)
# convert the prediction to 0, 1 with 0.5 threshold
# model_predict > 0.5 returns a boolean array, astype('int32') converts True to 1 and False to 0
predict_class = (model_predict > 0.5).astype('int32') 
conf_matrix = confusion_matrix(y_test, predict_class)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
target_names = ['Reject', 'Approve']
classification_rep = classification_report(y_test, predict_class)



# endregion ========= Functional API Model Evaluation =========

