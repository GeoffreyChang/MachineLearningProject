import multiprocessing
import warnings
from itertools import repeat
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from keras import metrics
from helper_functions import *
import numpy as np
import tensorflow as tf
from keras.backend import clear_session
from itertools import chain

# Ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Get the number of CPU cores
n_pools = multiprocessing.cpu_count()


# Define a function to run the model on a single k-fold of the data
def run_cv_fold(feat, targ, train_index, test_index, epoch, batch_no):
    # Split the features and target into training and validation sets
    x_tra, x_val = feat.iloc[train_index], feat.iloc[test_index]
    y_tra, y_val = targ.iloc[train_index], targ.iloc[test_index]

    # Convert the data to numpy arrays
    x_tra = np.array(x_tra)
    x_val = np.array(x_val)

    # Reshape the data for the LSTM model
    x_tra = np.reshape(x_tra, (x_tra.shape[0], 1, x_tra.shape[1]))
    x_val = np.reshape(x_val, (x_val.shape[0], 1, x_val.shape[1]))

    # Reset the Keras session
    clear_session()

    # Build the GRU model
    model_run = tf.keras.Sequential()
    model_run.add(tf.keras.layers.GRU(64, input_shape=(x_tra.shape[1], x_tra.shape[2])))
    model_run.add(tf.keras.layers.Dense(1))
    model_run.compile(
        loss='mean_squared_error',
        metrics=[metrics.RootMeanSquaredError()],
        optimizer=tf.keras.optimizers.Adam(0.001)
    )

    # evaluate the model on the test data
    model_run.fit(x_tra, y_tra, epochs=epoch, batch_size=batch_no, verbose=0)

    # Predict the values
    predicted_val = model_run.predict(x_val)
    return predicted_val, y_val


def main(epoch, batch_no, kfold_splits):
    df = read_all_files()
    df.pop(9)  # remove the 10th file due to missing data
    df.pop(5)  # remove the 6th file due to extremely high values
    # concatenate all the dataframes into a single dataframe
    df = pd.concat(df)
    df = df.drop(["TIME", "S"], axis=1)
    df.dropna(inplace=True)
    df_norm = normalize_df(df)
    df_norm.reset_index(drop=True, inplace=True)

    # Split the data into features and target
    features, target = get_features_and_target(df_norm)

    # use k-fold cross validation to evaluate the model
    kfold = KFold(n_splits=kfold_splits, shuffle=True, random_state=42)

    # run the model for each fold as a multiprocessing task
    a, b = [train_ind for train_ind, _ in kfold.split(features)], [val_ind for _, val_ind in kfold.split(features)]
    with multiprocessing.Pool(n_pools) as pool:
        result = pool.starmap(run_cv_fold, zip(repeat(features), repeat(target), a, b, repeat(epoch), repeat(batch_no)))

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

    # print the results
    print("--------------------------------------")
    print('Average scores for all folds:')
    print(f'Mean R^2 After Denormilisation: {np.mean(total_r2)}')
    print(f'Mean RMSE Before Denormilisation: {np.mean(total_rmse)}')
    print("--------------------------------------")

    # Plot the predicted vs real values
    z_plot(predicted_overall, real_overall, no_epochs=epoch, batch_no=batch_no, no_kfold=kfold_splits,
           split=False, save_fig=__file__.split("\\")[-1][:-3])


if __name__ == "__main__":
    main(epoch=100, batch_no=16, kfold_splits=10)