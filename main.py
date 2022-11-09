import pandas as pd
import seaborn as sns
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb
from sklearn.model_selection import train_test_split

(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=10000)
print(x_train)
path = os.getcwd()
path = os.path.join(path, "Dataset/")
excel_files = glob.glob(os.path.join(path, "*.xlsx"))

if __name__ == "__main__":
    main_df = []
    for f in excel_files:
        df = pd.read_excel(f)
        main_df.append(df)
    main_df = pd.concat(main_df)
    normalized_df=(main_df-main_df.min())/(main_df.max()-main_df.min())
    # k_fold = KFold(n_splits=10)
    # for train_indices, test_indices in k_fold.split(normalized_df):
    #     print('Train: %s | test: %s' % (train_indices, test_indices))
    #     X_train, X_test = normalized_df[train_indices], normalized_df[test_indices]
    #     Y_train, Y_test = normalized_df[train_indices], normalized_df[test_indices]
    cols = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12']
    X = normalized_df[cols]
    Y = normalized_df['Z']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)
    maxlen = 200
    batch_size = 128
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    y_train = np.array(y_train)
    y_test = np.array(y_test)


