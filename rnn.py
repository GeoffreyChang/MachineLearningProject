import tensorflow as tf
from sklearn.model_selection import KFold
from helper_functions import *

folds = KFold(n_splits=10)


if __name__ == "__main__":
    df = read_file_no(1)
    features, target = get_features_and_target(df)

    scores = []
    for train_index, test_index in folds.split(df):
        x_train, x_test, y_train, y_test = features.iloc[train_index], \
                                           features.iloc[test_index], \
                                           target.iloc[train_index], \
                                           target.iloc[test_index]


