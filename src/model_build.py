from tensorflow import keras

def build_model(input_shape):
    input_ = keras.layers.Input(shape=(input_shape,))
    hidden1 = keras.layers.Dense(64, activation='relu')(input_)
    hidden2 = keras.layers.Dense(32, activation='relu')(hidden1)
    hidden3 = keras.layers.Dense(16, activation='relu')(hidden2)
    output_ = keras.layers.Dense(1, activation='sigmoid')(hidden3)
    model = keras.Model(inputs=input_, outputs=output_)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model