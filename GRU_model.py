from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from helper_functions import *
import numpy as np
from keras import layers, Sequential

folds = KFold(n_splits=5)


if __name__ == "__main__":
    df = read_all_files(2)
    df = normalize_df(df)

    df = df.sample(frac=1, random_state=1)
    features, target = get_features_and_target(df)
    features = series_to_supervised(features, 1, 1)
    predicted_overall = []
    real_overall = []
    for train_index, test_index in folds.split(features):
        x_train, x_test, y_train, y_test = features.iloc[train_index], \
                                           features.iloc[test_index], \
                                           target.iloc[train_index], \
                                           target.iloc[test_index]
        # keras.backend.clear_session()
        model_gru = Sequential()
        model_gru.add(layers.GRU(64, input_shape=(features.shape[1], 1)))
        model_gru.add(layers.Dense(1))
        model_gru.compile(
            loss='mean_absolute_error',
            optimizer="adam"
        )

        y_hat = fit_model(model_gru, x_train, y_train, x_test, y_test, epoc=10, name="GRU")

        predicted_overall.append(y_hat)
        real_overall.append(y_test)
        z_plot(y_hat, y_test)
        # z_plot_comparison(y_hat, y_test)
        # residuals_plot(y_hat, y_test)
        # break
        rmse = sqrt(mean_squared_error(y_test, y_hat))

        print('Test RMSE: %.3f' % rmse)

    z_plot(predicted_overall, real_overall, split=True)