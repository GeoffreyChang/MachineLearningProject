import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns

def read_all_files(n=None):
    """
        Plots all predicted vs real values of Z
        :param n: optinal file no to read
        :return: df: returns respective pandas dataframe
        """
    path = os.getcwd()
    path = os.path.join(path, "dataset/")
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    csv_files.sort(key=lambda i: int(os.path.splitext(os.path.basename(i))[0][8:]))
    if n:
        df = pd.read_csv(csv_files[n - 1])
        df.rename(columns={df.columns[0]: 'TIME', df.columns[1]: 'S'}, inplace=True)
        return df
    df = []
    for f in csv_files:
        try:
            data = pd.read_csv(f)
            data.rename(columns={data.columns[0]: 'TIME', data.columns[1]: 'S'}, inplace=True)
            df.append(data)
        except ValueError:
            pass
    return df

def entire_dataset():
    return pd.concat(read_all_files())

def get_features_and_target(df):
    features, target = df.iloc[:,:-1], df["Z"]
    features = features.drop(["TIME", "S"], axis=1)
    return features, target

def z_plot(predicted, real, split=True):
    """
    Plots all predicted vs real values of Z
    :param predicted: list of lists of predicted values
    :param real: list of lists of true values
    :param split: if True, plots each k-fold separately
    :return: None
    """
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    if not isinstance(predicted, list):
        ax.plot(real, predicted, 'k.', alpha=0.5)
        ax.plot([real.min(), real.max()], [real.min(), real.max()], 'r--')
        overall_score = r2_score(real, predicted)

    elif split:
        overall_score = []
        for i, (r, p) in enumerate(zip(real, predicted)):
            ax.plot(r, p, '.', alpha=0.5, label=f'Fold {i+1}')
            overall_score.append(r2_score(r, p))
            ax.plot([r.min(), r.max()], [r.min(), r.max()], 'r-')
        ax.legend()
        overall_score = np.mean(overall_score)
    else:
        real_all = [inner for outer in real for inner in outer]
        predicted_all = [inner for outer in predicted for inner in outer]
        ax.plot(real_all, predicted_all, 'k.', alpha=0.3)
        overall_score = np.mean(r2_score(real_all, predicted_all))
        ax.plot([min(real_all), max(real_all)], [min(real_all), max(real_all)], 'r--')

    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('R2: {:.3f}'.format(overall_score))
    plt.show()

def find_best_features(df):
    try:
        df = df.drop(["TIME", "S"], axis=1)
    except KeyError:
        pass

    priority_queue = []
    names = df.corr()["Z"].sort_values(ascending=False).index
    corr = df.corr()["Z"].sort_values(ascending=False)
    for n, c in zip(names, corr):
        if c >= 0.4:
            priority_queue.append(n)

    priority_queue.pop(0)
    selected_features = []
    discarded_features = []
    for i, col in enumerate(df.columns[:-1]):
        if col not in discarded_features:
            corr = list(df.corr()[col])
            names = list(df.corr().index)
            corr.pop(i), names.pop(i)
            high_corr = [col]
            for n, c in zip(names, corr):
                if c >= 0.9 and n != "Z" and n not in discarded_features:
                    high_corr.append(n)
            for stuff in priority_queue:
                if stuff in high_corr and stuff:
                    if stuff in selected_features:
                        pass
                    else:
                        selected_features.append(high_corr.pop(high_corr.index(stuff)))
                    break
            discarded_features.extend(high_corr)
    return selected_features

def normalize_df(df):
    return (df - df.min()) / (df.max() - df.min())

def residuals_plot(predicted, real):
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    residual = predicted - real
    sns.residplot(x = predicted, y = residual, data = None, scatter_kws={'color': 'r',
   'alpha': 0.5}, ax=ax)
    ax.set_xlabel('Predicted Value')
    ax.set_ylabel('Residuals')
    plt.show()

def series_to_supervised(data, n_in=1, n_out=1, drop_nan=True):
     """
     Frame a time series as a supervised learning dataset.
     Arguments:
     data: Sequence of observations as a list or NumPy array.
     n_in: Number of lag observations as input (X).
     n_out: Number of observations as output (y).
     drop_nan: Boolean whether to drop rows with NaN values.
     Returns:
     Pandas DataFrame of series framed for supervised learning.
     """
     n_vars = 1 if type(data) is list else data.shape[1]
     df = pd.DataFrame(data)
     cols, names = list(), list()
     # input sequence (t-n, ... t-1)
     for i in range(n_in, 0, -1):
         cols.append(df.shift(i))
         names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
     # forecast sequence (t, t+1, ... t+n)
     for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
     # put it all together
     agg = pd.concat(cols, axis=1)
     agg.columns = names
     # drop rows with NaN values
     if drop_nan:
        agg.dropna(inplace=True)
     return agg

def fit_model(model, x_train, y_train, x_test, y_test, epoc, name):
    """
    Fits a model to the training data and returns the predicted values
    :param model: model to fit
    :param x_train: training data
    :param y_train: training labels
    :param x_test: test features
    :param y_test: test targets
    :param epoc: number of epochs to train for
    :param name: name of the model
    :return: predicted values
    """
    model.fit(x_train, y_train, epochs=epoc)
    y_hat = model.predict(x_test)
    print(f"Model: {name} | R2: {r2_score(y_test, y_hat)}")
    return y_hat