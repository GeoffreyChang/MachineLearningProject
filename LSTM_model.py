import multiprocessing
from itertools import repeat
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from keras import metrics
from helper_functions import *
import numpy as np
import tensorflow as tf
from keras.backend import clear_session
n_pools = multiprocessing.cpu_count()

folds = KFold(n_splits=6)
# from keras import mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)


def run_cv_fold(feat, targ, train_index, test_index):
    # split the data into training and testing sets for this fold
    x_tra , x_val = feat.iloc[train_index], feat.iloc[test_index]
    y_tra, y_val = targ.iloc[train_index], targ.iloc[test_index]

    clear_session()
    model_run = tf.keras.Sequential()
    model_run.add(tf.keras.layers.LSTM(64, input_shape=(feat.shape[1], 1)))
    model_run.add(tf.keras.layers.Dense(64, activation='relu'))
    model_run.add(tf.keras.layers.Dense(64))
    model_run.add(tf.keras.layers.Dense(1))
    model_run.compile(
        loss='mean_absolute_error',
        metrics=[metrics.RootMeanSquaredError()],
        optimizer=tf.keras.optimizers.Adam(0.001)
    )

    # evaluate the model on the test data
    model_run.fit(x_tra, y_tra, epochs=30, batch_size=16, verbose=1)
    Y_hat = model_run.predict(x_val)
    return Y_hat, y_val


if __name__ == "__main__":
    # df = read_all_files(1)
    df = read_all_files()
    df.pop(9)
    df.pop(5)
    df = pd.concat(df)
    df = df.drop(["TIME", "S"], axis=1)
    df = normalize_df(df)
    # df = df.sample(frac=1, random_state=1)
    # features, target = get_features_and_target(df)
    df = df.dropna()
    features, target = df.iloc[:, :-1], df["Z"]
    # features = normalize_df(features)
    scores = []
    predicted_overall = []
    real_overall = []

    kfold = KFold(n_splits=7, shuffle=True, random_state=1)  # use 7-fold cross validation
    a = []
    b = []
    for train_ind, val_ind in kfold.split(features):
        a.append(train_ind)
        b.append(val_ind)

    a, b = [train_ind for train_ind, _ in kfold.split(features)], [val_ind for _, val_ind in kfold.split(features)]

    result = multiprocessing.Pool(n_pools).starmap(run_cv_fold, zip(repeat(features), repeat(target), a, b))

    for predic, real in result:
        predicted_overall.append(predic)
        real_overall.append(real)

    # calculate the mean squared error
    total_rmse = []
    for i in range(len(predicted_overall)):
        rmse = tf.sqrt(tf.reduce_mean(tf.math.squared_difference(predicted_overall[i], real_overall[i])))
        # print('Fold %d: %.3f' % (i + 1, rmse))
        total_rmse.append(rmse)
    print('Mean RMSE: %.3f' % np.mean(total_rmse))

    z_plot(predicted_overall, real_overall, False)