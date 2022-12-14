from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from helper_functions import *
import numpy as np
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import metrics
import multiprocessing
from itertools import repeat
import warnings
from keras.backend import clear_session
warnings.filterwarnings("ignore", category=FutureWarning)

n_pools = multiprocessing.cpu_count()

def run_cv_fold(feat, targ, train_index, test_index):
    # split the data into training and testing sets for this fold
    x_tra , x_val = feat[train_index], feat[test_index]
    y_tra, y_val = targ[train_index], targ[test_index]
    clear_session()
    model_run = Sequential()
    model_run.add(LSTM(64, input_shape=(feat[0].shape[0], feat[0].shape[1])))  # 64 units in the LSTM layer
    model_run.add(Dense(1))  # output layer with one unit
    model_run.compile(
        optimizer='adam',
        loss='mse',
        metrics=[metrics.RootMeanSquaredError()])

    # evaluate the model on the test data
    model_run.fit(x_tra, y_tra, epochs=50, batch_size=16, verbose=1)
    Y_hat = model_run.predict(x_val)
    return Y_hat, y_val

if __name__ == "__main__":
    window_size = 3
    # load dataset [df1, df2, ..., df14]
    data = read_all_files()
    # remove df10 (missing values from T9 and T10)
    data.pop(9)
    # remove df6 (extremely high z-axis displacement)
    data.pop(5)
    data = pd.concat(data)
    # split into input and output variables
    data = data.drop(["TIME", "S"], axis=1)

    data = normalize_df(data)

    # Standardize the temperature data

    data = data.dropna()
    X, y = data.iloc[:, :-1], data["Z"]
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(X)

    # use the sliding window method to create sequences
    # for example, if window_size = 3, the sequences will be
    # [[X[0], X[1], X[2]], [X[1], X[2], X[3]], ..., [X[n-3], X[n-2], X[n-1]]]
    # and the corresponding output will be [y[2], y[3], ..., y[n-1]]
    # choose a window size
    X_seq = []
    y_seq = []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size])
        y_seq.append(y[i + window_size])

    # convert to numpy arrays
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    predicted_overall = []
    real_overall = []



    # use k-fold cross validation to evaluate the model
    kfold = KFold(n_splits=7, shuffle=True, random_state=1)  # use 7-fold cross validation
    a = []
    b = []
    for train_ind, val_ind in kfold.split(X_seq):
        a.append(train_ind)
        b.append(val_ind)

    a, b = [train_ind for train_ind, _ in kfold.split(X_seq)], [val_ind for _, val_ind in kfold.split(X_seq)]

    result = multiprocessing.Pool(n_pools).starmap(run_cv_fold, zip(repeat(X_seq), repeat(y_seq), a, b))

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
