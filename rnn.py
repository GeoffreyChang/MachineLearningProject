import tensorflow as tf
from keras.layers import Embedding, Dense, LSTM
from keras.losses import BinaryCrossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import KFold
from keras.datasets import imdb
folds = KFold(n_splits=10)

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == "__main__":
    df = pd.read_excel("Dataset/Thermal expansion testing data 01.xlsx")

    # Model configuration
    additional_metrics = ['accuracy']
    batch_size = 128
    embedding_output_dims = 15
    loss_function = BinaryCrossentropy()
    max_sequence_length = 300
    num_distinct_words = 5000
    number_of_epochs = 10
    optimizer = Adam()
    validation_split = 0.40
    verbosity_mode = 1

    # Disable eager execution
    tf.compat.v1.disable_eager_execution()

    df.drop(columns=df.columns[[0, 1]], axis=1, inplace=True)
    df.columns = [''] * len(df.columns)
    features = df.copy()
    target = df.iloc[:, -1]
    features.drop(columns=features.columns[[-1]], axis=1, inplace=True)
    features = features.dropna(axis=1)

    for train_index, test_index in folds.split(df):
        x_train, x_test, y_train, y_test = features.iloc[[i for i in train_index]], features.iloc[[i for i in test_index]], \
                                           target.iloc[[i for i in train_index]], target.iloc[[i for i in test_index]]

        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_distinct_words)

        # Pad all sequences
        padded_inputs = pad_sequences(x_train, maxlen=max_sequence_length, value=0.0)
        padded_inputs_test = pad_sequences(x_test, maxlen=max_sequence_length, value=0.0)

        # Define the Keras model
        model = Sequential()
        model.add(Embedding(num_distinct_words, embedding_output_dims, input_length=max_sequence_length))
        model.add(LSTM(10))
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(optimizer=optimizer, loss=loss_function, metrics=additional_metrics)

        # Give a summary
        model.summary()

        # Train the model
        history = model.fit(padded_inputs, y_train, batch_size=batch_size, epochs=number_of_epochs, verbose=verbosity_mode,
                            validation_split=validation_split)

        # Test the model after training
        test_results = model.evaluate(padded_inputs_test, y_test, verbose=False)
        print(f'Test results - Loss: {test_results[0]} - Accuracy: {100 * test_results[1]}%')
        break