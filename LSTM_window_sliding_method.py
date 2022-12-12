from helper_functions import *
import numpy as np
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras import metrics
import multiprocessing
from itertools import repeat

n_pools = multiprocessing.cpu_count()

def run_cv_fold(model_run, feat, targ, train_index, test_index):
    # split the data into training and testing sets for this fold
    x_tra , x_val = feat[train_index], feat[test_index]
    y_tra, y_val = targ[train_index], targ[test_index]

    # evaluate the model on the test data
    model_run.fit(x_tra, y_tra, epochs=50, batch_size=64, verbose=0)
    Y_hat = model_run.predict(x_val)
    return Y_hat, y_val

if __name__ == "__main__":
    window_size = 3
    # load dataset
    data = read_all_files()
    data.pop(9)
    data.pop(5)
    data = pd.concat(data)
    # data = data.sample(frac=1, random_state=1)
    # split into input and output variables
    data = data.drop(["TIME", "S"], axis=1)
    data = normalize_df(data)
    data = data.dropna()
    X, y = data.iloc[:, :-1], data["Z"]
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
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

    model = Sequential()
    model.add(LSTM(20, input_shape=(window_size, X.shape[1])))  # 20 units in the LSTM layer
    model.add(Dense(1))  # output layer with one unit
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=[metrics.RootMeanSquaredError()])

    # use k-fold cross validation to evaluate the model
    kfold = KFold(n_splits=5, shuffle=False)  # use 5-fold cross validation
    a = []
    b = []
    for train_ind, val_ind in kfold.split(X_seq):
        a.append(train_ind)
        b.append(val_ind)

    # c, d = zip(*[(train_ind, val_ind) for train_ind, val_ind in kfold.split(X_seq)])
    a, b = [train_ind for train_ind, _ in kfold.split(X_seq)], [val_ind for _, val_ind in kfold.split(X_seq)]

    result = multiprocessing.Pool(n_pools).starmap(run_cv_fold, zip(repeat(model), repeat(X_seq), repeat(y_seq), a, b))
    # residuals_plot(y_hat, y_test)
    for predic, real in result:
        predicted_overall.append(predic)
        real_overall.append(real)
    z_plot(predicted_overall, real_overall, False)
