from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from yellowbrick.regressor import ResidualsPlot
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

folds = KFold(n_splits=10)


if __name__ == "__main__":
    df = pd.read_excel("Dataset/Thermal expansion testing data 01.xlsx")
    df.drop(columns=df.columns[[0, 1]], axis=1, inplace=True)
    df = (df - df.min()) / (df.max() - df.min())
    features = df.copy()
    target = df.iloc[:, -1]
    features.drop(columns=features.columns[[-1]], axis=1, inplace=True)
    features = features.dropna(axis=1)

    for train_index, test_index in folds.split(df):
        X_train, X_test, Y_train, Y_test = features.iloc[[i for i in train_index]], features.iloc[[i for i in test_index]], \
                                           target.iloc[[i for i in train_index]], target.iloc[[i for i in test_index]]

        # X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.3)

        # scaler = MinMaxScaler(feature_range=(0, 1))
        # X_train = scaler.fit_transform(X_train)
        # X_train = pd.DataFrame(X_train)
        # X_test = scaler.fit_transform(X_test)
        # X_test = pd.DataFrame(X_test)

        # regressor = SVR(kernel='rbf')
        # regressor.fit(X_train, Y_train)
        # regressor.score(X_test, Y_test)

        # Implement SVR with Hyper-Parameter Tuning

        # K = 15
        # parameters = [
        #     {'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9], 'C': [1, 10, 100, 1000, 10000]}]
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
        model = SVR(kernel='rbf')
        model.fit(X_train, Y_train)
        print(model.score(X_test, Y_test))

        plt.figure(figsize=(10, 6))
        visualizer = ResidualsPlot(model)
        visualizer.fit(X_train, Y_train)
        visualizer.score(X_test, Y_test)
        visualizer.show()