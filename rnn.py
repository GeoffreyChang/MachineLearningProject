import tensorflow as tf
from keras.layers import Embedding, Dense, LSTM
from keras.losses import BinaryCrossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras_preprocessing.sequence import pad_sequences
from tensorflow import keras
from keras import layers
import pandas as pd
from sklearn.model_selection import KFold
folds = KFold(n_splits=10)
from keras import metrics
QUALITY_THRESHOLD = 128
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 2


def create_model():
    input_layer = keras.Input(shape=(12, 1))

    x = layers.Conv1D(
        filters=32, kernel_size=3, strides=2, activation="relu", padding="same"
    )(input_layer)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(
        filters=64, kernel_size=3, strides=2, activation="relu", padding="same"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(
        filters=128, kernel_size=5, strides=2, activation="relu", padding="same"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(
        filters=256, kernel_size=5, strides=2, activation="relu", padding="same"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(
        filters=512, kernel_size=7, strides=2, activation="relu", padding="same"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(
        filters=1024, kernel_size=7, strides=2, activation="relu", padding="same"
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(4096, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(
        2048, activation="relu", kernel_regularizer=keras.regularizers.L2()
    )(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(
        1024, activation="relu", kernel_regularizer=keras.regularizers.L2()
    )(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(
        128, activation="relu", kernel_regularizer=keras.regularizers.L2()
    )(x)
    output_layer = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs=input_layer, outputs=output_layer)


if __name__ == "__main__":
    df = pd.read_excel("Dataset/Thermal expansion testing data 01.xlsx")
    df.drop(columns=df.columns[[0, 1]], axis=1, inplace=True)

    features = df.copy()
    target = df.iloc[:, -1]
    features.drop(columns=features.columns[[-1]], axis=1, inplace=True)
    features = features.dropna(axis=1)
    num_classes = len(features.columns)
    for train_index, test_index in folds.split(df):
        x_train, x_test, y_train, y_test = features.iloc[[i for i in train_index]], features.iloc[[i for i in test_index]], \
                                           target.iloc[[i for i in train_index]], target.iloc[[i for i in test_index]]

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        test_dataset = test_dataset.batch(BATCH_SIZE)
        conv_model = create_model()

        epochs = 30

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                "best_model.h5", save_best_only=True, monitor="loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_top_k_categorical_accuracy",
                factor=0.2,
                patience=2,
                min_lr=0.000001,
            ),
        ]

        optimizer = keras.optimizers.Adam(amsgrad=True, learning_rate=0.001)
        loss = keras.losses.CategoricalCrossentropy()

        conv_model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

        conv_model_history = conv_model.fit(
            train_dataset,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=test_dataset
        )
