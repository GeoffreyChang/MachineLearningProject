import multiprocessing
import warnings
from itertools import repeat, chain
import tensorflow as tf
from keras.backend import clear_session
from sklearn.model_selection import KFold, TimeSeriesSplit
from tensorflow import keras
from helper_functions import *
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.style.use('ggplot')

n_pools = multiprocessing.cpu_count()
epoch = 100
batch_no = 16
kfold_splits = 10

def build_and_compile_model():
    clear_session()
    model = keras.Sequential([
        keras.layers.Dense(64, activation='sigmoid'),
        keras.layers.Dense(64),
        keras.layers.Dense(1)])

    model.compile(
        loss='mean_squared_error',
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
        optimizer=tf.keras.optimizers.Adam(0.001))
    return model


def run_cv_fold(feat, targ, train_index, test_index):
    # split the data into training and testing sets for this fold
    x_tra, x_val = feat.iloc[train_index], feat.iloc[test_index]
    y_tra, y_val = targ.iloc[train_index], targ.iloc[test_index]

    model_run = build_and_compile_model()

    # evaluate the model on the test data
    model_run.fit(x_tra, y_tra, epochs=epoch, batch_size=batch_no, verbose=0)

    Y_hat = model_run.predict(x_val)
    return Y_hat, y_val


def main():
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
        df_norm.reset_index(inplace=True, drop=True)
        features, target = df_norm.iloc[:, :-1], df_norm["Z"]

    kfold = KFold(n_splits=kfold_splits, shuffle=True, random_state=42)

    a, b = [train_ind for train_ind, _ in kfold.split(features)], [val_ind for _, val_ind in kfold.split(features)]
    with multiprocessing.Pool(n_pools) as pool:
        result = pool.starmap(run_cv_fold, zip(repeat(features), repeat(target), a, b))

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
    print('Average scores for all folds:')
    print(f'Mean R^2: {np.mean(total_r2)}')
    print(f'Mean RMSE: {np.mean(total_rmse)}')
    print("--------------------------------------")
    z_plot(predicted_overall, real_overall, split=False)

if __name__ == '__main__':
    main()