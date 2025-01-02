import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os
import datetime

def load_data():
    """
    Loads the loan assessment data from the specified source.

    Returns:
        DataFrame: A pandas DataFrame containing the loan assessment data.
    """
    raw_data = pd.read_csv('./data/raw_loan_data.csv')
    mata_data = pd.read_csv('./data/metadata.txt', sep='\t', header=None, names=['feature', 'description', 'type'])
    return raw_data, mata_data

def preprocess_data(raw_data):
    """
    Preprocesses the loan assessment data.

    Args:
        raw_data (DataFrame): A pandas DataFrame containing the loan assessment data.

    Returns:
        DataFrame: A pandas DataFrame containing the preprocessed loan assessment data.
    """
    # feature management
    num_features = raw_data.select_dtypes(exclude=['object']).columns.tolist()
    cat_features = raw_data.select_dtypes(include=['object']).columns.tolist()

    # split data
    X = raw_data.drop(columns=['loan_status'], axis=1)
    y = raw_data['loan_status']
    X_temp, X_test, y_tempt, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_tempt, test_size=0.25, random_state=42)
    
    # Preprocessing for numerical data
    num_transformer = StandardScaler()

    # Preprocessing for categorical data
    cat_transformer = OrdinalEncoder()

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ])

    # Preprocess the data
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(input_shape, model_name):
    """
    Builds a Sequential model according to the specified input shape.

    Args:
        input_shape (tuple): A tuple containing the input shape.

    Returns:
        Model: A Sequential model.
    """
    # Define the model
    input_ = Input(shape=(input_shape,), name='input_layer')
    hidden1 = Dense(64, activation='relu', name='hidden_layer_1')(input_)
    hidden2 = Dense(32, activation='relu', name='hidden_layer_2')(hidden1)
    hidden3 = Dense(16, activation='relu', name='hidden_layer_3')(hidden2)
    output_ = Dense(1, activation='sigmoid', name='output_layer')(hidden3)
    model = keras.Model(inputs=input_, outputs=output_, name=model_name)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, model_name):
    """
    Trains the specified model.

    Args:
        model (Model): A Sequential model.
        X_train (DataFrame): A pandas DataFrame containing the training data.
        y_train (DataFrame): A pandas DataFrame containing the training labels.
        X_val (DataFrame): A pandas DataFrame containing the validation data.
        y_val (DataFrame): A pandas DataFrame containing the validation labels.
        model_name (str): A string containing the model name.
    """
    # Define the callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_checkpoint = ModelCheckpoint(filepath=f'./model_save/{model_name}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.keras', monitor='val_loss', save_best_only=True, save_weights_only=False, save_freq='epoch')
    log_dir = "./logs/fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', embeddings_freq=0, embeddings_metadata=None)

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=32, callbacks=[early_stopping, model_checkpoint, tensorboard])

    return history

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the specified model.

    Args:
        model (Model): 
        X_test (DataFrame): A pandas DataFrame containing the test data.
        y_test (DataFrame): A pandas DataFrame containing the test labels.
    """
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {test_loss}')
    print(f'Test accuracy: {test_accuracy}')
    

def predict(model, X):
    """
    Predicts the labels of the specified data.

    Args:
        model (Model): A Sequential model.
        X (DataFrame): A pandas DataFrame containing the data.

    Returns:
        DataFrame: A pandas DataFrame containing the predicted labels.
    """
    # Predict the labels
    y_pred = model.predict(X)
    return y_pred

def main():
    # Load the data
    raw_data, meta_data = load_data()

    # Preprocess the data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(raw_data)

    # Build the model
    model = build_model(X_train.shape[1], 'loan_assessment_model')

    # Train the model
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, model_name='loan_assessment_model')

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Predict the labels
    y_pred = predict(model, X_test)

if __name__ == '__main__':
    main()