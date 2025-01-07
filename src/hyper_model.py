from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner
import datetime

# Define the model architecture
def model_architecture(input_shape, units, activation, num_layers, optimizers, lr):
    """
    Builds a Keras model with the specified hyperparameters.

    Args:
        input_shape (tuple): Shape of the input data.
        units (int): Number of units in each hidden layer.
        activation (str): Activation function for the hidden layers.
        num_layers (int): Number of hidden layers.
        optimizer (keras.optimizers.Optimizer): Optimizer for compiling the model.
        lr (float): Learning rate for the optimizer.

    Returns:
        keras.Model: Compiled Keras model.
    """
    # Define the input layer
    input = keras.Input(shape=input_shape)
    
    # Add hidden layers
    x = input
    for i in range(num_layers):
        x = layers.Dense(units=units, activation=activation, name=f"hidden_layer_{i+1}")(x)
    
    # Define the output layer
    output_status = layers.Dense(units=1, activation='sigmoid', name='output_status')(x)
    output_score = layers.Dense(units=1, activation='linear', name='output_score')(x)

    # Create the model
    model = keras.Model(inputs=input,
                        outputs=[output_status, output_score],
                        name='tuned_model')
    
    # Select the optimizer
    if optimizers == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=lr)
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss={'output_status': 'binary_crossentropy',
              'output_score': 'mean_squared_error'
              },
        
        metrics={
            'output_status': ['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')],
            'output_score': ['mean_squared_error']})

    return model

# Define the hyperparameters
hp = keras_tuner.HyperParameters()
def build_hyper_model(hp):
    """
    Builds a hyperparameter-tuned Keras model.

    Args:
        input_shape (tuple): Shape of the input data.
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
    input_shape = (12,)

    # Build the model
    model = model_architecture(input_shape, units, activation, num_layers, optimizers, lr)

    return model


# Define the callbacks
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='model_checkpoint.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

early_stop_callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    mode='min'
)
