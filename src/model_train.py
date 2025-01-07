import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import datetime

def create_callbacks():
    """
    Creates and returns a list of Keras callbacks for model training.
    This function creates three callbacks:
    1. TensorBoard callback: Logs training metrics for visualization in TensorBoard.
    2. ModelCheckpoint callback: Saves the model or weights at specified intervals during training.
    3. EarlyStopping callback: Stops training when a monitored metric has stopped improving.
    Returns:
        list: A list containing the TensorBoard, ModelCheckpoint, and EarlyStopping callbacks.
    """
    # tensorboard callback
    log_dir = f"./logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', embeddings_freq=0, embeddings_metadata=None)
    
    # model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath=f'./model_save/full_model_checkpoint_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.keras',
        save_weights_only=False,
        save_freq='epoch',
        monitor='val_loss',
        save_best_only=True,)
    
    # early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights=True)
    
    return [tensorboard_callback, checkpoint_callback, early_stopping_callback]

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    """
    Trains the given model using the provided training and validation data.

    Args:
        model (tf.keras.Model): The model to be trained.
        X_train (numpy.ndarray): Training data features.
        y_train (numpy.ndarray): Training data labels.
        X_val (numpy.ndarray): Validation data features.
        y_val (numpy.ndarray): Validation data labels.
        epochs (int, optional): Number of epochs to train the model. Defaults to 10.
        batch_size (int, optional): Number of samples per gradient update. Defaults to 32.

    Returns:
        None
    """
    callbacks = create_callbacks()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
    # Save the final model after training
    model.save(f'./model_save/end_training_model_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.keras')