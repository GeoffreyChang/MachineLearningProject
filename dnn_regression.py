import multiprocessing
import warnings
from itertools import repeat

import numpy as np
import tensorflow as tf
from keras.backend import clear_session
from sklearn.model_selection import KFold
from tensorflow import keras

from helper_functions import *

warnings.simplefilter(action='ignore', category=FutureWarning)
n_pools = multiprocessing.cpu_count()
plt.style.use('ggplot')


def build_and_compile_model():
    clear_session()
    model = keras.Sequential([
        keras.layers.Dense(64, activation='sigmoid'),
        keras.layers.Dense(64),
        keras.layers.Dense(1)])

    model.compile(loss='mean_absolute_error',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()],
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


def run_cv_fold(feat, targ, train_index, test_index):
    # split the data into training and testing sets for this fold
    x_tra, x_val = feat.iloc[train_index], feat.iloc[test_index]
    y_tra, y_val = targ.iloc[train_index], targ.iloc[test_index]

    model_run = build_and_compile_model()

    # evaluate the model on the test data
    model_run.fit(x_tra, y_tra, epochs=50, batch_size=16, verbose=1)

    Y_hat = model_run.predict(x_val)
    return Y_hat, y_val


if __name__ == "__main__":
    separate = False

    if separate:
        df = read_all_files(1)
        df_norm = normalize_df(df)
        best_features = find_best_features(df_norm)
        # df_norm = df_norm.sample(frac=1, random_state=1)
        features, target = get_features_and_target(df_norm)
        features = features[best_features]
    else:
        df = read_all_files()
        df.pop(9)
        df.pop(5)
        df = pd.concat(df)
        # df = df.sample(frac=1, random_state=12)
        df = df.drop(["TIME", "S"], axis=1)
        df_norm = normalize_df(df)
        df_norm = df_norm.dropna()
        features, target = df_norm.iloc[:, :-1], df_norm["Z"]

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    a = []
    b = []
    predicted_overall = []
    real_overall = []
    for train_ind, val_ind in kfold.split(features):
        a.append(train_ind)
        b.append(val_ind)
        # hold_target.append(target1.iloc[val_ind])

    a, b = [train_ind for train_ind, _ in kfold.split(features)], [val_ind for _, val_ind in kfold.split(features)]
    with multiprocessing.Pool(n_pools) as pool:
        result = pool.starmap(run_cv_fold, zip(repeat(features), repeat(target), a, b))

    total_rmse = []
    for predic, real in result:
        predicted_overall.append(predic)
        real_overall.append(real)
        rmse_metric = tf.keras.metrics.RootMeanSquaredError()
        rmse_metric.update_state(y_true=real, y_pred=predic)
        total_rmse.append(rmse_metric.result().numpy())

    print('Mean RMSE: %.3f' % np.mean(total_rmse))
    z_plot(predicted_overall, real_overall, split=False)
