import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import datetime

def create_callbacks():
    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', embeddings_freq=0, embeddings_metadata=None)
    checkpoint_callback = ModelCheckpoint(
        filepath=f'./model_save/full_model_checkpoint_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.keras',
        save_weights_only=False,
        save_freq='epoch')
    return [tensorboard_callback, checkpoint_callback]

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    callbacks = create_callbacks()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
    model.save('./model_save/loan_assessment_model.keras')