import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import datetime

def load_data(raw_data_path, metadata_path):
    raw_data = pd.read_csv(raw_data_path)
    metadata = pd.read_csv(metadata_path, sep='\t', header=None, names=['Feature', 'Description', 'Type'])
    num_features = raw_data.select_dtypes(exclude='object').columns.tolist()
    cat_features = raw_data.select_dtypes(include='object').columns.tolist()
    return raw_data, num_features, cat_features

def split_data(raw_data):
    X = raw_data.drop(['loan_status'], axis=1)
    y = raw_data['loan_status']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_data(X_train, X_val, X_test, num_features, cat_features):
    col_transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), [feature for feature in num_features if feature != 'loan_status']),
            ('cat', OrdinalEncoder(), cat_features)], remainder='passthrough')
    X_train_transform = col_transformer.fit_transform(X_train)
    X_val_transform = col_transformer.transform(X_val)
    X_test_transform = col_transformer.transform(X_test)
    return X_train_transform, X_val_transform, X_test_transform

def create_callbacks():
    log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', embeddings_freq=0, embeddings_metadata=None, profile_batch=2)
    checkpoint_callback = ModelCheckpoint(
        filepath=f'./model_save/full_model_checkpoint_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.keras',
        save_weights_only=False,
        save_freq='epoch')
    return [tensorboard_callback, checkpoint_callback]

def build_model(input_shape):
    input_ = keras.layers.Input(shape=(input_shape,))
    hidden1 = keras.layers.Dense(64, activation='relu')(input_)
    hidden2 = keras.layers.Dense(32, activation='relu')(hidden1)
    hidden3 = keras.layers.Dense(16, activation='relu')(hidden2)
    output_ = keras.layers.Dense(1, activation='sigmoid')(hidden3)
    model = keras.Model(inputs=input_, outputs=output_)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    raw_data_path = './data/raw_loan_data.csv'
    metadata_path = './data/metadata.txt'
    raw_data, num_features, cat_features = load_data(raw_data_path, metadata_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(raw_data)
    X_train_transform, X_val_transform, X_test_transform = preprocess_data(X_train, X_val, X_test, num_features, cat_features)
    callbacks = create_callbacks()
    model = build_model(X_train_transform.shape[1])
    tf.profiler.experimental.start(callbacks[0].log_dir)
    model.fit(X_train_transform, y_train, epochs=10, batch_size=32, validation_data=(X_val_transform, y_val), callbacks=callbacks, verbose=1)
    tf.profiler.experimental.stop()
    model.evaluate(X_test_transform, y_test)
    model_predict = model.predict(X_test_transform)
    return model, model_predict

if __name__ == "__main__":
    main()
