import os
import time
import datetime
import tensorflow as tf
import keras_tuner
from keras.utils import plot_model
import matplotlib.pyplot as plt
from src.data_loader import load_data
from src.data_preprocessing import split_data, preprocess_data, transform_features
from src.model_train import *
from src.hpmodel import *
random_state = 42
# Data loader
start_time = time.time()
raw_data_path = './data/raw_loan_data.csv'
metadata_path = './data/metadata.txt'
raw_data, num_features, cat_features, metadata = load_data(raw_data_path, metadata_path)
print(f"Data loading time: {time.time() - start_time:.2f} seconds")

# original indices
original_indices = raw_data.index

# Data split
start_time = time.time()
X_train, X_val, X_test, y_train_loan_status, y_val_loan_status, y_test_loan_status, y_train_score, y_val_score, y_test_score = split_data(raw_data)

# outlier detection
from sklearn.ensemble import IsolationForest
outlier_features = ['person_age', 'person_emp_exp', 'cb_person_cred_hist_length']
outlier_detector = IsolationForest(contamination=0.007, random_state=random_state)
X_train['outliers'] = outlier_detector.fit_predict(X_train[outlier_features])
indices_to_remove = X_train[X_train['outliers'] == -1].index

# Remove outliers
X_train.drop(index=indices_to_remove, columns='outliers', inplace=True)
y_train_loan_status.drop(labels=indices_to_remove, inplace=True)
y_train_score.drop(labels=indices_to_remove, inplace=True)

# Remove outliers from validation set
X_val['outliers'] = outlier_detector.predict(X_val[outlier_features])
val_indices_to_remove = X_val[X_val['outliers'] == -1].index
X_val.drop(index=val_indices_to_remove, columns='outliers', inplace=True)
y_val_loan_status.drop(labels=val_indices_to_remove, inplace=True)
y_val_score.drop(labels=val_indices_to_remove, inplace=True)

# Remove outliers from test set
X_test['outliers'] = outlier_detector.predict(X_test[outlier_features])
test_indices_to_remove = X_test[X_test['outliers'] == -1].index
X_test.drop(index=test_indices_to_remove, columns='outliers', inplace=True)
y_test_loan_status.drop(labels=test_indices_to_remove, inplace=True)
y_test_score.drop(labels=test_indices_to_remove, inplace=True)



# Plot box plot
# fig, axs = plt.subplots(3, 1, figsize=(12, 12))
# for i, feature in enumerate(X_train[['person_age', 'person_emp_exp', 'cb_person_cred_hist_length']].columns):
#     ax = axs[i]
#     boxplot = ax.boxplot([X_train[X_train['outliers'] == 1][feature], X_train[X_train['outliers'] == -1][feature]], 
#                          labels=['Inliers', 'Outliers'], vert=False, showmeans=True, meanline=True)
#     ax.set_title(f'Box Plot of {feature}', fontweight='bold', fontsize=14)
#     ax.set_xlabel(feature)
#     ax.set_ylabel(f'Outliers Detection')
    
#     # Add mean annotation for inliers
#     mean_inliers = boxplot['means'][0].get_xdata()[0]
#     ax.annotate(f'Mean: {mean_inliers:.2f}', xy=(mean_inliers, 1), xytext=(mean_inliers + 0.5, 1.1),
#                 horizontalalignment='center', verticalalignment='center')
    
#     # Add mean annotation for outliers
#     mean_outliers = boxplot['means'][1].get_xdata()[0]
#     ax.annotate(f'Mean: {mean_outliers:.2f}', xy=(mean_outliers, 2), xytext=(mean_outliers + 1, 2.1),
#                 horizontalalignment='center', verticalalignment='center')
    
#     # Draw vertical line through min values
#     # min_inliers = boxplot['whiskers'][0].get_xdata()[0]
#     min_outliers = boxplot['caps'][2].get_xdata()[0]
#     # ax.axvline(min_inliers, color='r', linestyle='--', label='Min Inliers')
#     ax.axvline(min_outliers, color='r', linestyle='--', label='Min Outliers')
    
#     # Add tick label for red line
#     # ax.annotate(f'{min_inliers:.2f}', xy=(min_inliers, 0.5), xytext=(min_inliers, 0.5),
#     #             color='red', horizontalalignment='center', verticalalignment='center')
#     ax.annotate(f'{min_outliers:.2f}', xy=(min_outliers, 1.5), xytext=(min_outliers, 1.5),
#                 color='red', horizontalalignment='center', verticalalignment='center')
    
#     ax.legend()

# fig.tight_layout()
# plt.show()

# plt.figure(figsize=(12, 8))
# plt.scatter(X_train['person_age'], X_train['person_emp_exp'], alpha=0.5)
# plt.title('Scatter Plot of Person Age vs Employment Experience', fontweight='bold', fontsize=14)
# plt.xlabel('Person Age')
# plt.ylabel('Employment Experience')
# plt.show()


# plt.figure(figsize=(12, 8))
# scatter = plt.scatter(X_train['cb_person_cred_hist_length'], X_train['person_age'], c=X_train['loan_intent'].astype('category').cat.codes, alpha=0.5, cmap='viridis')
# plt.title('Scatter Plot of Credit History Length vs Person Age', fontweight='bold', fontsize=14)
# plt.xlabel('Credit History Length')
# plt.ylabel('Person Age')
# plt.colorbar(scatter, label='Loan Intent')

# # Add horizontal line at 60
# plt.axhline(y=60, color='r', linestyle='--', label='Age 60')

# # Add legend
# loan_intent_labels = X_train['loan_intent'].astype('category').cat.categories
# handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i / len(loan_intent_labels)), markersize=10) for i in range(len(loan_intent_labels))]
# handles.append(plt.Line2D([0], [0], color='r', linestyle='--'))  # Add horizontal line to legend
# plt.legend(handles, list(loan_intent_labels) + ['Age 60'], title='Loan Intent')

# plt.show()

# Data preprocessing
X_train_transform, X_val_transform, X_test_transform, y_train_score_transform, y_val_score_transform, y_test_score_transform, col_transformer, y_scaler = preprocess_data(
    X_train, X_val, X_test, y_train_score, y_val_score, y_test_score, num_features, cat_features)

# keep track features for interpretate
transform_features = transform_features(col_transformer)
print(f"Data preprocessing time: {time.time() - start_time:.2f} seconds")

#  plot features after transform
import matplotlib.style
fig, axs = plt.subplots(4, 3, figsize=(24, 18))
matplotlib.style.use('seaborn-v0_8-whitegrid') # Set global theme style

for i, feature in enumerate(transform_features):
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
    max_trials=30,
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

# tuner.search(
#     X_train, y=[y_train_loan_status, y_train_score],
#     validation_data=(X_val, [y_val_loan_status, y_val_score]), epochs=10,
#     callbacks=callbacks
# )

print(f"Hyperparameter tuning time: {time.time() - start_time:.2f} seconds")


# Retrieve and save the top 10 best models
save_dir = 'save_models/best'
os.makedirs(save_dir, exist_ok=True)
best_models = tuner.get_best_models(num_models=10)[0]
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

