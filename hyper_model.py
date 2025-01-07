from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner

def model_architechture(input_shape, units, activation, num_layers, optimizers, lr):
    """
    Builds a Keras model with the specified hyperparameters.

    Args:
        units (int): Number of units in each hidden layer.
        activation (str): Activation function for the hidden layers.
        num_layers (int): Number of hidden layers.
        optimizer (keras.optimizers.Optimizer): Optimizer for compiling the model.
        lr (float): Learning rate for the optimizer.

    Returns:
        keras.Model: Compiled Keras model.
    """
    #  Define the input layer
    input = keras.Input(shape=(input_shape,))
    
    #  add hidden layers
    x = input
    for i in range(num_layers):
        x = layers.Dense(units=units, activation=activation, name=f"hidden_layer_{i+1}")(x)
    
    # Define the output layer
    output = layers.Dense(units=1, activation='sigmoid', name='output_layer')(x)

    # Create the model
    model = keras.Model(inputs=input, outputs=output, name='tuned_model')
    
    # select the optimizer
    if optimizers == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=lr)
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    return model

def build_hyper_model(hp):
    """
    Builds a hyperparameter-tuned Keras model.

    Args:
        input_shape (int): Number of features in the input data.
        hp (keras_tuner.engine.hyperparameters.HyperParameters): HyperParameters object
            containing hyperparameters to tune.

    Returns:
        keras.Model: Hyperparameter-tuned Keras model.
    """
    # Define the hyperparameters
    num_layers = hp.Int('num_layers', min_value=2, max_value=5, default=3)
    units = hp.Int('units', min_value=32, max_value=256, step=32, default=64)
    activation = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'], default='relu')
    optimizers = hp.Choice('optimizers', values=['adam', 'sgd'], default='adam')
    lr = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
    input_shape = 13    

    # Build the model
    model = model_architechture(input_shape, units, activation, num_layers, optimizers, lr)

    return model

hp = keras_tuner.HyperParameters()
model = build_hyper_model(hp)

# create tuner
tuner = keras_tuner.GridSearch(
    hypermodel=build_hyper_model, # model builder function
    objective=keras_tuner.Objective('val_accuracy', direction='max'), # metric to optimize
    max_trials=10, # number of hyperparameter combinations to try
    directory='hyperparameter_search',
    project_name='loan_default_prediction',
    overwrite=True # overwrite the results of the previous run save in directory
)




from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import numpy as np
import datetime

def model_architecture(input_shape, units, activation, num_layers, optimizers, lr):
    """
    Builds a Keras model with the specified hyperparameters. 
    This is model architecture function that is used in the GridSearch tuner.
    Take arguments from build_hyper_model function and build the model.

    Args:
        units (int): Number of units in each hidden layer.
        activation (str): Activation function for the hidden layers.
        num_layers (int): Number of hidden layers.
        optimizer (keras.optimizers.Optimizer): Optimizer for compiling the model.
        lr (float): Learning rate for the optimizer.

    Returns:
        keras.Model: Compiled Keras model.
    """
    # Define the input layer
    input = keras.Input(shape=(input_shape,))
    
    # Add hidden layers
    x = input
    for i in range(num_layers):
        x = layers.Dense(units=units, activation=activation, name=f"hidden_layer_{i+1}")(x)
    
    # Define the output layer
    output = layers.Dense(units=1, activation='sigmoid', name='output_layer')(x)

    # Create the model
    model = keras.Model(inputs=input, outputs=output, name='tuned_model')
    
    # Select the optimizer
    if optimizers == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=lr)
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def build_hyper_model(hp):
    """
    Builds a hyperparameter-tuned Keras model. that is used in the GridSearch tuner.
    set the hyperparameters to tune. 
    Hyperparameters are arguments for model_architecture function.

    Args:
        hp (keras_tuner.engine.hyperparameters.HyperParameters): HyperParameters object
            containing hyperparameters to tune.

    Returns:
        keras.Model: Hyperparameter-tuned Keras model.
    """
    # Define the hyperparameters (search space)
    num_layers = hp.Int('num_layers', min_value=2, max_value=5, default=3)
    units = hp.Int('units', min_value=32, max_value=256, step=32, default=64)
    activation = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'], default='relu')
    optimizers = hp.Choice('optimizers', values=['adam', 'sgd'], default='adam')
    lr = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
    input_shape = 13    

    # Build the model
    model = model_architecture(input_shape, units, activation, num_layers, optimizers, lr)

    return model

# Create the GridSearch tuner
tuner = kt.GridSearch(
    hypermodel=build_hyper_model,
    objective='val_accuracy',
    max_trials=20,  # Number of different hyperparameter sets to try
    executions_per_trial=1,  # Number of times to execute training for each set of hyperparameters
    directory='my_dir',
    project_name='grid_search'
)

# Define the callbacks
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='model_checkpoint.h5',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

early_stop_callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    mode='min'
)