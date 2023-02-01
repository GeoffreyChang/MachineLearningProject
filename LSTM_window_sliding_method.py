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
import time


def run_cv_fold(feat, targ, train_index, test_index, epoch, batch_no, num_units=64, learning_rate=0.001):
    # split the data into training and testing sets for this fold
    x_tra , x_val = feat[train_index], feat[test_index]
    y_tra, y_val = targ[train_index],   targ[test_index]

    clear_session()
    model_run = Sequential()
    model_run.add(Bidirectional(LSTM(num_units, input_shape=(feat[0].shape[0], feat[0].shape[1]))))  # 64 units in the LSTM layer
    model_run.add(Dense(1))  # output layer with one unit
    model_run.compile(
        loss='mean_squared_error',
        metrics=[metrics.RootMeanSquaredError()],
        optimizer=tf.keras.optimizers.Adam(learning_rate)
    )

    # evaluate the model on the test data
    model_run.fit(x_tra, y_tra, epochs=epoch, batch_size=batch_no, verbose=1)
    predicted_val = model_run.predict(x_val)
    return predicted_val, y_val


def main(epoch, batch_no, kfold_splits, window_size, num_units=64, learning_rate=0.001):
    df = read_all_files()
    df.pop(11) # remove the 12th file as it is identical to the 11th file
    df.pop(9) # remove the 10th file due to missing data
    df.pop(5) # remove the 6th file due to extremely high values

    df = pd.concat(df)
    df = df.drop(["TIME", "S"], axis=1)
    df.dropna(inplace=True)
    df_norm = normalize_df(df)
    df_norm.reset_index(drop=True, inplace=True)

    x, y = get_features_and_target(df_norm)

    # use the sliding window method to create sequences
    # for example, if window_size = 3, the sequences will be
    # [[X[0], X[1], X[2]], [X[1], X[2], X[3]], ..., [X[n-3], X[n-2], X[n-1]]]
    # and the corresponding output will be [y[2], y[3], ..., y[n-1]]
    # choose a window size
    x_seq = []
    y_seq = []
    for i in range(len(x) - window_size):
        x_seq.append(x[i:i + window_size])
        y_seq.append(y[i + window_size])

    # convert to numpy arrays
    x_seq = np.array(x_seq)
    y_seq = np.array(y_seq)




    # use k-fold cross validation to evaluate the model
    kfold = KFold(n_splits=kfold_splits, shuffle=True, random_state=1)

    a, b = [train_ind for train_ind, _ in kfold.split(x_seq)], [val_ind for _, val_ind in kfold.split(x_seq)]
    start_time = time.time()
    with multiprocessing.Pool(n_pools) as pool:
        result = pool.starmap(run_cv_fold, zip(repeat(x_seq), repeat(y_seq), a, b, repeat(epoch), repeat(batch_no),
                                               repeat(num_units), repeat(learning_rate)))

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

    total_time = time.time() - start_time
    print("--------------------------------------")
    print('Average scores for all folds after denormilisation:')
    print(f"Total Time: {total_time} seconds")
    print(f'Mean R^2: {np.mean(total_r2)}')
    print(f'Mean RMSE: {np.mean(total_rmse)}')
    print("--------------------------------------")
    # record_results(__file__, np.mean(total_rmse), np.mean(total_r2), epoch, batch_no, kfold_splits, total_time)

    # Plot the predicted vs real values
    z_plot(predicted_overall, real_overall, no_epochs=epoch, batch_no=batch_no, no_kfold=kfold_splits,
           split=False)
    # save_fig = __file__.split("\\")[-1][:-3]
    return np.mean(total_rmse)
if __name__ == "__main__":
    main(epoch=100, batch_no=32, kfold_splits=10, window_size=28, num_units=64, learning_rate=0.001)
