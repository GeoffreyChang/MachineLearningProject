import pandas as pd
import seaborn as sns
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

folds = StratifiedKFold(n_splits=5)

path = os.getcwd()
path = os.path.join(path, "Dataset/")
excel_files = glob.glob(os.path.join(path, "*.xlsx"))


def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)
    pass


def binary_decision_tree(X):
    X = (X - X.min()) / (X.max() - X.min())

    y = list(X.iloc[:, -1])
    lab = preprocessing.LabelEncoder()
    # y = lab.fit_transform(y)
    X.drop(columns=X.columns[[0, 1, -1]], axis=1, inplace=True)
    X = X.dropna(axis=1)

    clf_score = []
    svm_score = []
    random_score = []

    for train_index, test_index in folds.split(X, y):
        X_train, X_test, y_train, y_test = X.iloc[[train_index]].values, X.iloc[[test_index]].values, y[[train_index]], y[[test_index]]
        print(get_score(tree.DecisionTreeClassifier(max_depth=5), X_train, X_test, y_train, y_test))
        # print(get_score(SVC(), X_train, X_test, y_train, y_test))
        # print(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))
        print()

    # clf = tree.DecisionTreeClassifier(max_depth=5)
    # clf.fit(X_train, y_train)
    #
    # title = f.split("\\")[-1][26:-5]
    # print(f"DecisionTree - R² for {title}: ", "%.2f" % clf.score(X_test, y_test))
    #
    # svm = SVC()
    # svm.fit(X_train, y_train)
    # title = f.split("\\")[-1][26:-5]
    # print(f"SVM - R² for {title}: ", "%.2f" % svm.score(X_test, y_test))
    #
    # rf = RandomForestClassifier(n_estimators=40)
    # rf.fit(X_train, y_train)
    # title = f.split("\\")[-1][26:-5]
    # print(f"Random Forest - R² for {title}: ", "%.2f" % rf.score(X_test, y_test))



if __name__ == "__main__":
    main_df = []
    for i, f in enumerate(excel_files):
        df = pd.read_excel(f)
        binary_decision_tree(df)
        break
