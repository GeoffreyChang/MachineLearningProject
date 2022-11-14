import pandas as pd
import seaborn as sns
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt


folds = KFold(n_splits=10)
n_bins = 5

path = os.getcwd()
path = os.path.join(path, "Dataset/")
excel_files = glob.glob(os.path.join(path, "*.xlsx"))


def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


def ml_model(X):
    # normilize the data
    # X = (X - X.min()) / (X.max() - X.min())

    qt = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')

    X.drop(columns=X.columns[[0, 1]], axis=1, inplace=True)

    features = X.copy()
    target = X.iloc[:, -1]
    lab = preprocessing.LabelEncoder()
    # y = lab.fit_transform(y)
    features.drop(columns=features.columns[[-1]], axis=1, inplace=True)
    features = features.dropna(axis=1)


    scaled_feature_names = [f"BIN_{x}" for x in X]
    X[scaled_feature_names] = qt.fit_transform(X)
    X[scaled_feature_names] = X[scaled_feature_names].astype(int)


    # clf_score = []
    # svm_score = []
    # random_score = []

    target = X[scaled_feature_names].iloc[:, -1]
    features = X[scaled_feature_names].copy()
    features.drop(columns=features.columns[[-1]], axis=1, inplace=True)

    # for train_index, test_index in folds.split(X[scaled_feature_names]):
    #     X_train, X_test, y_train, y_test = features.iloc[[i for i in train_index]], features.iloc[[i for i in test_index]], \
    #                                        target.iloc[[i for i in train_index]], target.iloc[[i for i in test_index]]
    #
    #     clf_score.append(get_score(tree.DecisionTreeClassifier(), X_train, X_test, y_train, y_test))
    #     svm_score.append(get_score(SVC(), X_train, X_test, y_train, y_test))
    #     random_score.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))

    print(cross_val_score(tree.DecisionTreeClassifier(max_depth=5), features, target))
    print(cross_val_score(SVC(), features, target))
    print(cross_val_score(RandomForestClassifier(n_estimators=40), features, target))




if __name__ == "__main__":
    main_df = []
    for i, f in enumerate(excel_files):
        df = pd.read_excel(f)
        ml_model(df)
        break
