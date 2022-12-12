from sklearn.model_selection import KFold
from helper_functions import *
import numpy as np
import tensorflow as tf

folds = KFold(n_splits=6)
# from keras import mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

if __name__ == "__main__":
    df = read_all_files(1)
    df = normalize_df(df)
    df = df.sample(frac=1, random_state=1)
    features, target = get_features_and_target(df)
    scores = []
    predicted_overall = []
    real_overall = []

    for train_index, test_index in folds.split(features):
        x_train, x_test, y_train, y_test = features.iloc[train_index], \
                                           features.iloc[test_index], \
                                           target.iloc[train_index], \
                                           target.iloc[test_index]
        # keras.backend.clear_session()
        model_lstm = tf.keras.Sequential()
        model_lstm.add(tf.keras.layers.LSTM(64, input_shape=(features.shape[1], 1)))
        model_lstm.add(tf.keras.layers.Dense(64, activation='relu'))
        model_lstm.add(tf.keras.layers.Dense(64))
        model_lstm.add(tf.keras.layers.Dense(1))
        model_lstm.compile(
            loss='mean_absolute_error',
            optimizer=tf.keras.optimizers.Adam(0.001)
        )

        y_hat = fit_model(model_lstm, x_train, y_train, x_test, y_test, epoc=10, name="LSTM")

        predicted_overall.append(y_hat)
        real_overall.append(y_test)
        # residuals_plot(y_hat, y_test)

    z_plot(predicted_overall, real_overall)
    print(np.mean(scores))