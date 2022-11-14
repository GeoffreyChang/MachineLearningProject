from numpy import array
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.model_selection import KFold

folds = KFold(n_splits=10)

if __name__ == "__main__":
    df = pd.read_excel("Dataset/Thermal expansion testing data 01.xlsx")
    df = (df - df.min()) / (df.max() - df.min())
    df.drop(columns=df.columns[[0, 1]], axis=1, inplace=True)
    features = df.copy()
    target = df.iloc[:, -1]
    features.drop(columns=features.columns[[-1]], axis=1, inplace=True)
    features = features.dropna(axis=1)





    for train_index, test_index in folds.split(df):
        x_train, x_test, y_train, y_test = features.iloc[[i for i in train_index]], features.iloc[[i for i in test_index]], \
                                           target.iloc[[i for i in train_index]], target.iloc[[i for i in test_index]]
        tf.keras.backend.clear_session()
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(12, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        model.summary()

        model.fit(x_train, y_train, epochs=5, verbose=1)

        print("Evaluate on test data")
        results = model.evaluate(x_test, y_test)
        print("test loss, test acc:", results)
        break