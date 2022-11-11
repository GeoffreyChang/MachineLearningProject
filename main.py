import pandas as pd
import seaborn as sns
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree


path = os.getcwd()
path = os.path.join(path, "Dataset/")
excel_files = glob.glob(os.path.join(path, "*.xlsx"))

def binary_decision_tree(X):
    X = (X - X.min()) / (X.max() - X.min())

    y = list(X.iloc[:, -1])
    lab = preprocessing.LabelEncoder()
    y = lab.fit_transform(y)
    X.drop(columns=X.columns[[0, 1, -1]], axis=1, inplace=True)
    X = X.dropna(axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)

    title = f.split("\\")[-1][26:-5]
    print(f"R² for {title}: ", "%.2f" % clf.score(X_test, y_test))


if __name__ == "__main__":
    main_df = []
    for i, f in enumerate(excel_files):
        df = pd.read_excel(f)
        binary_decision_tree(df)