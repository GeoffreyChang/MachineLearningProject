from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn import metrics
import math
import matplotlib.pyplot as plt
from yellowbrick.regressor import ResidualsPlot
folds = KFold(n_splits=10)
plt.style.use('ggplot')
if __name__ == "__main__":
    df = pd.read_excel("Dataset/Thermal expansion testing data 01.xlsx")
    df.drop(columns=df.columns[[0, 1]], axis=1, inplace=True)

    features = df.copy()
    target = df.iloc[:, -1]
    features.drop(columns=features.columns[[-1]], axis=1, inplace=True)
    features = features.dropna(axis=1)


    for train_index, test_index in folds.split(df):
        x_train, x_test, y_train, y_test = features.iloc[[i for i in train_index]], \
                                           features.iloc[[i for i in test_index]], \
                                           target.iloc[[i for i in train_index]], \
                                           target.iloc[[i for i in test_index]]
        model = Ridge(alpha=1.0)
        model.fit(x_train, y_train)

        # Testing
        y_hat = model.predict(x_test)
        print("R square: %.3f" % r2_score(y_test, y_hat))

        # plt.figure(figsize=(10, 6))
        # visualizer = ResidualsPlot(model)
        # visualizer.fit(x_train, y_train)
        # visualizer.score(x_test, y_test)
        # visualizer.show()
        fig, ax = plt.subplots()
        ax.plot(y_test, y_hat, 'b.')
        ax.plot([y_test.min(), y_test.max()], [y_hat.min(), y_hat.max()], 'r--')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('R2: {:.3f}'.format(r2_score(y_test, y_hat)))
        plt.show()