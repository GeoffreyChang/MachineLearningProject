from sklearn.model_selection import KFold
from helper_functions import *
import numpy as np
from keras import layers
import keras

folds = KFold(n_splits=6)


if __name__ == "__main__":
    df = read_file_no(2)
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
        model_lstm = keras.Sequential()
        model_lstm.add(layers.GRU(64, input_shape=(features.shape[1], 1)))
        model_lstm.add(layers.Dense(1))
        model_lstm.compile(
            loss='mean_absolute_error',
            optimizer="adam"
        )

        model_lstm.fit(x_train, y_train, epochs=10, verbose=1)

        y_hat = model_lstm.predict(x_test).flatten()
        predicted_overall.append(y_hat)
        real_overall.append(y_test)
        score = r2_score(y_test, y_hat)
        scores.append(score)
        print("R square: %.3f" % score)
        # z_plot_comparison(y_hat, y_test)
        # residuals_plot(y_hat, y_test)
        # break


    z_plot_all(predicted_overall, real_overall)
    print(np.mean(scores))