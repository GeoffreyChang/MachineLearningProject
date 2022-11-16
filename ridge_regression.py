from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from yellowbrick.regressor import ResidualsPlot
from sklearn.model_selection import cross_val_score
import numpy as np
from helper_functions import *

folds = KFold(n_splits=10)
plt.style.use('ggplot')


if __name__ == "__main__":
    files = read_all_files()
    # for df in files:
    df = pd.read_excel("Dataset/Thermal expansion testing data 07.xlsx")
    features, target = get_features_and_target(df)

    # print(np.mean(cross_val_score(Ridge(alpha=1.0), features, target, cv=10)))
    r_scores = []
    for train_index, test_index in folds.split(df):
        x_train, x_test, y_train, _ = features.iloc[[i for i in train_index]], \
                                           features.iloc[[i for i in test_index]], \
                                           target.iloc[[i for i in train_index]], \
                                           target.iloc[[i for i in test_index]]
        model = Ridge(alpha=1.0)
        model.fit(x_train, y_train)


        df2 = read_file_no(9)
        features2, target2 = get_features_and_target(df2)
        y_hat = model.predict(features2)
        y_test = target2

        # Testing
        # y_hat = model.predict(x_test)
        score = r2_score(y_test, y_hat)
        print("R square: %.3f" % score)
        r_scores.append(score)
        # plt.figure(figsize=(10, 6))
        # visualizer = ResidualsPlot(model)
        # visualizer.fit(x_train, y_train)
        # visualizer.score(x_test, y_test)
        # visualizer.show()

        # fig, ax = plt.subplots()
        # ax.plot(y_test, y_hat, 'b.')
        # ax.plot([y_test.min(), y_test.max()], [y_hat.min(), y_hat.max()], 'r--')
        # ax.set_xlabel('Actual')
        # ax.set_ylabel('Predicted')
        # ax.set_title('R2: {:.3f}'.format(r2_score(y_test, y_hat)))
        # plt.show()
