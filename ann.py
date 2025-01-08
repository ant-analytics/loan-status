import os
import time
import datetime
import tensorflow as tf
import keras_tuner
from keras.utils import plot_model
import matplotlib.pyplot as plt
from src.data_loader import load_data
from src.data_preprocessing import split_data, preprocess_data, X_features
from src.model_train import *
from src.hpmodel import *

# Data loader
start_time = time.time()
raw_data_path = './data/raw_loan_data.csv'
metadata_path = './data/metadata.txt'
raw_data, num_features, cat_features, metadata = load_data(raw_data_path, metadata_path)
print(f"Data loading time: {time.time() - start_time:.2f} seconds")

# Data preprocessing
start_time = time.time()
X_train, X_val, X_test, y_train_loan_status, y_val_loan_status, y_test_loan_status, y_train_score, y_val_score, y_test_score = split_data(raw_data)

X_train_transform, X_val_transform, X_test_transform, y_train_score_transform, y_val_score_transform, y_test_score_transform, col_transformer, y_scaler = preprocess_data(
    X_train, X_val, X_test, y_train_score, y_val_score, y_test_score, num_features, cat_features)

# keep track features for interpretate
X_features = X_features(col_transformer)
print(f"Data preprocessing time: {time.time() - start_time:.2f} seconds")

#  plot features after transform
import matplotlib.style
fig, axs = plt.subplots(4, 3, figsize=(24, 18))
matplotlib.style.use('seaborn-v0_8-whitegrid') # Set global theme style

for i, feature in enumerate(X_features):
    ax = axs[i//3, i%3]
    boxplot = ax.boxplot(X_test_transform[:, i], showfliers=True, vert=False, showmeans=True, meanline=True)
    description = metadata.loc[metadata['Feature'] == feature, 'Description'].values[0]
    ax.set_title(f'Box Plot of {description}', fontweight='bold', fontsize=14)
    ax.yaxis.set_ticklabels([])  # Set y-tick labels off
    
    # Add mean annotation
    # mean = boxplot['means'][0].get_xdata()[0]
    # ax.annotate(f'Mean: {mean:.2f}', xy=(mean, 1), xytext=(mean, 1.1),
    #             arrowprops=dict(facecolor='black', shrink=0.05),
    #             horizontalalignment='center', verticalalignment='center')
    
    # Add x-axis label
    ax.set_xlabel('Value')
    
    # Add legend for mean and median lines
    ax.legend([boxplot['means'][0], boxplot['medians'][0]], ['Mean', 'Median'], loc='upper right')

fig.tight_layout()
plt.show()

# Hyperparameter tuning
log_dir = f"logs/fit{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Create TensorBoard callback
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=log_dir,  
    histogram_freq=1,
    profile_batch='500,520'
)

early_stop_callback = keras.callbacks.EarlyStopping(
    monitor='val_output_status_accuracy',
    patience=3,
    mode='max'
)

callbacks = [tensorboard_callback, early_stop_callback]
 
tuner = keras_tuner.GridSearch(
    hypermodel=build_hyper_model,
    objective=[
        keras_tuner.Objective('val_output_status_accuracy', direction='max'),
        keras_tuner.Objective('val_output_score_mean_squared_error', direction='min')
    ],
    max_trials=50,
    directory=log_dir,
    project_name='loan_default_prediction',
    overwrite=True
)
# Tuning hyperparameters
start_time = time.time()
tuner.search(
    X_train_transform, y=[y_train_loan_status, y_train_score_transform],
    validation_data=(X_val_transform, [y_val_loan_status, y_val_score_transform]), epochs=10,
    callbacks=callbacks
)
print(f"Hyperparameter tuning time: {time.time() - start_time:.2f} seconds")


# Retrieve and save the top 10 best models
save_dir = 'save_models/best'
os.makedirs(save_dir, exist_ok=True)
best_models = tuner.get_best_models(num_models=10)
for i, model in enumerate(best_models):
    model.save(f'save_models/best/{i+1}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.keras')

save_worst = 'save_models/worst'
os.makedirs(save_worst, exist_ok=True)
worst_models = tuner.get_best_models(num_models=-10)
for i, model in enumerate(best_models):
    model.save(f'save_models/worst/{i+1}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.keras')

# load the best model
import keras
best_models = keras.models.load_model('save_models/best/1_20250107-232302.keras')

worst_models = keras.models.load_model('save_models/worst/1_20250107-232519.keras')

