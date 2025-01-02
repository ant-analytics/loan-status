from src.data_loader import load_data
from src.data_preprocessing import *
from src.model_build import *
from src.model_train import *


raw_data_path = './data/raw_loan_data.csv'
metadata_path = './data/metadata.txt'
raw_data, num_features, cat_features, metadate = load_data(raw_data_path, metadata_path)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(raw_data)
X_train_transform, X_val_transform, X_test_transform = preprocess_data(X_train, X_val, X_test, num_features, cat_features)
callbacks = create_callbacks()
model = build_model(X_train_transform.shape[1])
tf.profiler.experimental.start(callbacks[0].log_dir)
model.fit(X_train_transform, y_train, epochs=10, batch_size=32, validation_data=(X_val_transform, y_val), callbacks=callbacks, verbose=1)
tf.profiler.experimental.stop()
model.evaluate(X_test_transform, y_test)
model_predict = model.predict(X_test_transform)

