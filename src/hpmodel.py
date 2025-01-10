from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
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
    # scaler_choice = hp.Choice('scaler', values=['StandardScaler', 'MinMaxScaler', 'RobustScaler'], default='StandardScaler')

    # #  apply the chosen scaler
    # if scaler_choice == 'StandardScaler':
    #     scaler = StandardScaler()
    # elif scaler_choice == 'MinMaxScaler':
    #     scaler = MinMaxScaler()
    # else:
    #     scaler = RobustScaler()
    
    # # scale the data
    # X_train_transform = scaler.fit_transform(X_train)
    # X_val_transform = scaler.transform(X_val)

    # y_train_score_transform = scaler.fit_transform(y_train_score)
    # y_val_score_transform = scaler.transform(y_val_score)


    # Build the model
    input_shape = (12,)
    model = model_architecture(input_shape, units, activation, num_layers, optimizers, lr)

    return model