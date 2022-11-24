import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from yellowbrick.regressor import ResidualsPlot
from helper_functions import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
folds = KFold(n_splits=6)

plt.style.use('ggplot')
def build_and_compile_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='sigmoid'),
        keras.layers.Dense(64),
        keras.layers.Dense(1)])

    model.compile(loss='mean_absolute_error',
                optimizer=keras.optimizers.Adam(0.001))
    return model

if __name__ == "__main__":
    # normalizer = keras.layers.Normalization()
    files = read_all_files()
    main_scores = []


    # for df in files:
    df = read_file_no(1)
    df = normalize_df(df)
    best_features = find_best_features(df)
    # df = df[df.Z >= 9.5]
    df = df.sample(frac=1, random_state=1)
    features, target = get_features_and_target(df)

    features = features[best_features]
    scores = []
    dnn_model = build_and_compile_model()
    predicted_overall = []
    real_overall = []
    for train_index, test_index in folds.split(features):
        x_train, x_test, y_train, y_test = features.iloc[train_index], \
                                               features.iloc[test_index], \
                                           target.iloc[train_index], \
                                           target.iloc[test_index]
        history = dnn_model.fit(
            x_train,
            y_train,
            verbose=0,
            epochs=50)

        y_hat = dnn_model.predict(x_test).flatten()
        predicted_overall.append(y_hat)
        real_overall.append(y_test)
        scores.append(r2_score(y_test, y_hat))
        z_plot_comparison(y_hat, y_test)

    for s in scores:
        print("R square: %.3f" % s)
    main_scores.append(np.mean(scores))
    print("Average R square: %.3f" % np.mean(scores))
    z_plot_all(predicted_overall, real_overall)

    print("MAIN Average R square: %.3f" % np.mean(main_scores))
