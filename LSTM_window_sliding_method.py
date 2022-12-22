from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from helper_functions import *
import numpy as np
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
from keras import metrics
import multiprocessing
from itertools import repeat, chain
import warnings
from keras.backend import clear_session
warnings.filterwarnings("ignore", category=FutureWarning)

n_pools = multiprocessing.cpu_count()
epoch = 100
batch_no = 16
kfold_splits = 10


def run_cv_fold(feat, targ, train_index, test_index):
    # split the data into training and testing sets for this fold
    x_tra , x_val = feat[train_index], feat[test_index]
    y_tra, y_val = targ[train_index],   targ[test_index]
    clear_session()
    model_run = Sequential()
    model_run.add(Bidirectional(LSTM(64, input_shape=(feat[0].shape[0], feat[0].shape[1]))))  # 64 units in the LSTM layer
    model_run.add(Dense(1))  # output layer with one unit
    model_run.compile(
        loss='mean_squared_error',
        metrics=[metrics.RootMeanSquaredError()],
        optimizer=tf.keras.optimizers.Adam(0.001)
    )

    # evaluate the model on the test data
    model_run.fit(x_tra, y_tra, epochs=epoch, batch_size=batch_no, verbose=0)
    Y_hat = model_run.predict(x_val)
    return Y_hat, y_val

def main():
    window_size = 3
    # load dataset [df1, df2, ..., df14]
    df = read_all_files()
    # remove df10 (missing values from T9 and T10)
    df.pop(9)
    # remove df6 (extremely high z-axis displacement)
    df.pop(5)
    df = pd.concat(df)
    # split into input and output variables
    df = df.drop(["TIME", "S"], axis=1)
    df = df.dropna()
    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    # Standardize the temperature data

    df_norm.reset_index(drop=True)
    X, y = get_features_and_target(df_norm)

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


    # use k-fold cross validation to evaluate the model
    kfold = KFold(n_splits=kfold_splits, shuffle=True, random_state=1)  # use 7-fold cross validation

    a, b = [train_ind for train_ind, _ in kfold.split(X_seq)], [val_ind for _, val_ind in kfold.split(X_seq)]
    with multiprocessing.Pool(n_pools) as pool:
        result = pool.starmap(run_cv_fold, zip(repeat(X_seq), repeat(y_seq), a, b))

    total_rmse = []
    total_r2 = []
    predicted_overall = []
    real_overall = []
    for predic, real in result:
        # Denormalise the z-axis displacement
        predic = np.array(list(chain.from_iterable(predic)))
        predic = (pd.DataFrame(predic, columns=["Z"]) * (df.max() - df.min()) + df.min())["Z"]
        real = np.array(real)
        real = (pd.DataFrame(real, columns=["Z"]) * (df.max() - df.min()) + df.min())["Z"]

        # Append to overall list for plotting
        predicted_overall.append(predic)
        real_overall.append(real)

        # Calculate R^2 score
        r2 = r2_score(real, predic)
        total_r2.append(r2)

        # Calculate RMSE
        rmse_metric = tf.keras.metrics.RootMeanSquaredError()
        rmse_metric.update_state(y_true=real, y_pred=predic)
        total_rmse.append(rmse_metric.result().numpy())

    print("--------------------------------------")
    print('Average scores for all folds after denormilisation:')
    print(f'Mean R^2: {np.mean(total_r2)}')
    print(f'Mean RMSE: {np.mean(total_rmse)}')
    print("--------------------------------------")

    # Plot the predicted vs real values
    z_plot(predicted_overall, real_overall, split=False)

if __name__ == "__main__":
    main()
