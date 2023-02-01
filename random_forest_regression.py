from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
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
        df.pop(11)  # remove the 12th file as it is identical to the 11th file
        df.pop(9) # remove the 10th file due to missing data
        df.pop(5) # remove the 6th file due to extremely high values
        df = pd.concat(df)
        df = df.drop(["TIME", "S"], axis=1)
        df.dropna(inplace=True)
        df_norm = normalize_df(df)
        df_norm.reset_index(drop=True, inplace=True)
        features, target = get_features_and_target(df_norm)

    predicted_overall = []
    real_overall = []
    total_rmse = []
    total_r2 = []
    random_search = {'bootstrap': [True, False],
                    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                    'max_features': ['auto', 'sqrt'],
                    'min_samples_leaf': [1, 2, 4],
                    'min_samples_split': [2, 5, 10],
                    'n_estimators': [50, 100, 150, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
    grid_search = {
        'bootstrap': [False],
        'max_depth': [50, 60, 70, 80, 90, 100, None],
        'max_features': ['sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'n_estimators': list(range(50, 210, 10)),
        'random_state': [42]
    }

    param_grid = {'n_estimators': 50,
                  'min_samples_split': 2,
                  'min_samples_leaf': 1,
                  'max_features': 'sqrt',
                  'max_depth': 50,
                  'bootstrap': False,
                  'random_state': 42}

    # param_grid = {'n_estimators': 10,
    #               'min_samples_split': 2,
    #               'min_samples_leaf': 1,
    #               'max_features': 'sqrt',
    #               'max_depth': 13,
    #               'bootstrap': False,
    #               'random_state': 42}

    # results = open("graphs/z_plots/random_forest_regression/random_forest_regression_results.txt", 'r')
    # while True:
    #     line = results.readline()
    #     if not line:
    #         break
    #     param_grid = eval(line[:-1])
    for train_index, test_index in folds.split(df_norm):
        x_train, x_test, y_train, y_test = features.iloc[train_index], \
                                           features.iloc[test_index], \
                                           target.iloc[train_index], \
                                           target.iloc[test_index]
        model = RandomForestRegressor(**param_grid)
        # CV_rfr = GridSearchCV(estimator=RandomForestRegressor(), param_grid=grid, cv=5, verbose=1)
        # rf_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=5, verbose=2,
        #                                random_state=42, n_jobs=-1)
        model.fit(x_train, y_train)
        # record_results(__file__, CV_rfr.best_params_, 0, 0, 0, kfold_splits, 0)
        # print("BEST PARAMS: ", CV_rfr.best_params_)
        y_test.reset_index(drop=True, inplace=True)
        # Testing
        y_hat = model.predict(x_test)

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
           split=False, save_fig="")

if __name__ == "__main__":
    main()
