import time
from src.data_loader import *
from src.data_preprocessing import *
from src.model_train import *
from src.hyper_model import *
from keras.utils import plot_model
import math
import matplotlib.pyplot as plt
# data loader
start_time = time.time()
raw_data_path = './data/raw_loan_data.csv'
metadata_path = './data/metadata.txt'
raw_data, num_features, cat_features, metadata = load_data(raw_data_path, metadata_path)
print(f"Data loading time: {time.time() - start_time:.2f} seconds")

# data preprocessing
start_time = time.time()
X_train, X_val, X_test, y_train_loan_status, y_val_loan_status, y_test_loan_status, y_train_score, y_val_score, y_test_score = split_data(raw_data)

X_train_transform, X_val_transform, X_test_transform, y_train_score_transform, y_val_score_transform, y_test_score_transform, col_transformer, y_scaler = preprocess_data(X_train, X_val, X_test, y_train_score, y_val_score, y_test_score, num_features, cat_features)

print(f"Data preprocessing time: {time.time() - start_time:.2f} seconds")

# hyperparameter tuning
# generate tuner
log_dir = "logs_2/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tuner = keras_tuner.GridSearch(
    hypermodel=build_hyper_model, # model builder function
    objective=[
        keras_tuner.Objective('val_output_status_accuracy', direction='max'),
        keras_tuner.Objective('val_output_score_mean_squared_error', direction='min')
    ], # metrics to optimize
    max_trials=200, # number of hyperparameter combinations to try; should be larger enough to cover a wide range of hyperparameters combinations
    directory=log_dir,
    project_name='loan_default_prediction',
    overwrite=True # overwrite the results of the previous run save in directory
)


#  create tensorboard callback



# tuning hyperparameters
#  can add callbacks here as similar to fit function
start_time = time.time()
tuner.search(X_train_transform, y=[y_train_loan_status, y_train_score_transform],
             validation_data=(X_val_transform, [y_val_loan_status, y_val_score_transform]), epochs=20,
             callbacks = [keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),])
print(f"Hyperparameter tuning time: {time.time() - start_time:.2f} seconds")

# query tuner results
best_model = tuner.get_best_models(num_models=6)[0]
last_best_model = tuner.get_best_models(num_models=6)[-1]

# save the best model
best_model.save('best_model.keras')

# load the best model
# best_model = keras.models.load_model('best_model.keras')

# evaluate the best model
evaluation = best_model.evaluate(X_test_transform, [y_test_loan_status, y_test_score_transform])


# Print the evaluation results
for metric, value in zip(best_model.metrics_names, evaluation):
    print(f"{metric}: {value}")


# make predictions for top 2 samples
predictions = best_model.predict(X_test_transform[:2])
predicted_loan_status = predictions[0]
predicted_scores = predictions[1]

# print predictions for top 2 samples
print(f"true value of status: {y_test_loan_status[:2]}")
print("Predicted Loan Status for top 2 samples:", predicted_loan_status)

print(f"true value of scores: {y_test_score_transform[:2]}")
print("Predicted Scores for top 2 samples:", predicted_scores)

#  retrain the best model
def plot_hyperparameter_comparison(tuner, top_n=5):
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=top_n)
    metrics = ['val_output_status_accuracy', 'val_output_score_mean_squared_error']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        values = [hp.values[metric] for hp in best_hyperparameters]
        plt.plot(range(1, top_n + 1), values, marker='o')
        plt.title(f'Comparison of {metric} for Top {top_n} Models')
        plt.xlabel('Model Rank')
        plt.ylabel(metric)
        plt.grid(True)
        plt.show()

# Plot hyperparameter comparison for top 5 models
plot_hyperparameter_comparison(tuner, top_n=5)
