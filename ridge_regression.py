from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from keras.metrics import RootMeanSquaredError
from helper_functions import *
kfold_splits = 10
folds = KFold(n_splits=kfold_splits, shuffle=True, random_state=42)
plt.style.use('ggplot')


def main():
    separate = False

    if separate:
        df = read_all_files(1)
        df_norm = normalize_df(df)
        features, target = get_features_and_target(df_norm)
    else:
        df = read_all_files()
        df.pop(9) # remove the 10th file due to missing data
        df.pop(5) # remove the 6th file due to extremely high values
        df = pd.concat(df)
        df = df.drop(["TIME", "S"], axis=1)
        df.dropna(inplace=True)
        scaler = MinMaxScaler()
        df_norm = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        df_norm.reset_index(drop=True, inplace=True)
        features, target = get_features_and_target(df_norm)

    predicted_overall = []
    real_overall = []
    total_rmse = []
    total_r2 = []
    for train_index, test_index in folds.split(df_norm):
        x_train, x_test, y_train, y_test = features.iloc[train_index], \
                                           features.iloc[test_index], \
                                           target.iloc[train_index], \
                                           target.iloc[test_index]
        model = Ridge(alpha=1.0)
        model.fit(x_train, y_train)

        # Testing
        y_hat = model.predict(x_test)
        score = r2_score(y_test, y_hat)
        print("R square: %.3f" % score)

        # Denormalise the z-axis displacement
        predic = (pd.DataFrame(y_hat, columns=["Z"]) * (df.max() - df.min()) + df.min())["Z"]
        real = (pd.DataFrame(y_test, columns=["Z"]) * (df.max() - df.min()) + df.min())["Z"]

        # Append to overall list for plotting
        predicted_overall.append(predic)
        real_overall.append(real)

        # Calculate R^2 score
        r2 = r2_score(real, predic)
        total_r2.append(r2)

        # Calculate RMSE
        rmse_metric = RootMeanSquaredError()
        rmse_metric.update_state(y_true=real, y_pred=predic)
        total_rmse.append(rmse_metric.result().numpy())
    print("--------------------------------------")
    print('Average scores for all folds after denormilisation:')
    print(f'Mean R^2: {np.mean(total_r2)}')
    print(f'Mean RMSE: {np.mean(total_rmse)}')
    print("--------------------------------------")
    z_plot(predicted_overall, real_overall, no_kfold=kfold_splits,
           split=False, save_fig=__file__.split("\\")[-1][:-3])


if __name__ == "__main__":
    main()
