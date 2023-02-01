from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from helper_functions import *
folds = KFold(n_splits=10)


if __name__ == "__main__":
    separate = False

    if separate:
        df = read_all_files(1)
        df_norm = normalize_df(df)
        features1, target1 = get_features_and_target(df)
        features, target = get_features_and_target(df_norm)
    else:
        df = read_all_files()
        df.pop(9)
        df.pop(5)
        df = pd.concat(df)
        df = df.sample(frac=1, random_state=12)
        df = df.drop(["TIME", "S"], axis=1)
        df_norm = normalize_df(df)
        df_norm = df_norm.dropna()
        features, target = df_norm.iloc[:, :-1], df_norm["Z"]

    predicted_overall = []
    real_overall = []
    for train_index, test_index in folds.split(df_norm):
        x_train, x_test, y_train, y_test = features.iloc[train_index], \
                                           features.iloc[test_index], \
                                           target.iloc[train_index], \
                                           target.iloc[test_index]

        model = SVR(kernel='rbf')
        model.fit(x_train, y_train)
        # Testing
        y_hat = model.predict(x_test)
        predicted_overall.append(y_hat)
        real_overall.append(y_test)
        print("A")
        # Implement SVR with Hyper-Parameter Tuning

        # K = 15
        # parameters = [
        #     {'kernel': ['rbf'],
        #      'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9],
        #      'C': [1, 10, 100, 1000, 10000]}]
        # print("Tuning hyper-parameters")
        # scorer = make_scorer(mean_squared_error, greater_is_better=False)
        # svr = GridSearchCV(SVR(epsilon=0.01), parameters, cv=K, scoring=scorer)
        # svr.fit(X_train, Y_train)
        # print("Grid scores on training set:")
        # means = svr.cv_results_['mean_test_score']
        # stds = svr.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, svr.cv_results_['params']):
        #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        # model = SVR(kernel='rbf', C=1000, gamma=0.9)

    z_plot(predicted_overall, real_overall, split=False)