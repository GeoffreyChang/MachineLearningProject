from numpy import array
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.model_selection import KFold
from helper_functions import *
import numpy as np
from keras import layers
from tensorflow import keras

folds = KFold(n_splits=10)

def build_and_compile_model():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(None, 12)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
    return model


if __name__ == "__main__":
    df = read_file_no(1)

    df = df.sample(frac=1, random_state=1)
    features, target = get_features_and_target(df)
    scores = []
    for train_index, test_index in folds.split(df):
        x_train, x_test, y_train, y_test = features.iloc[train_index], \
                                           features.iloc[test_index], \
                                           target.iloc[train_index], \
                                           target.iloc[test_index]
        tf.keras.backend.clear_session()
        model_lstm = tf.keras.Sequential()
        model_lstm.add(layers.LSTM(64, input_shape=(12, 1)))
        model_lstm.add(layers.BatchNormalization())
        model_lstm.add(layers.Dense(1))
        model_lstm.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer="adam",
            metrics=["accuracy"],
        )
        model_lstm.fit(x_train, y_train, epochs=100, verbose=1)

        y_hat = model_lstm.predict(y_test).flatten()
        score = r2_score(y_test, y_hat)
        scores.append(score)
        print("R square: %.3f" % score)
        z_plot_comparison(y_hat, y_test)
    print(np.mean(scores))