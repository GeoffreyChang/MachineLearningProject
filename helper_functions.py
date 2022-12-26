import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
from keras.metrics import RootMeanSquaredError

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
    try:
        df = df.drop(["TIME", "S"], axis=1)
    except KeyError:
        pass
    features, target = df.iloc[:, :-1], df["Z"]
    return features, target

def z_plot(predicted, real, no_epochs=None, batch_no=None, no_kfold=None, split=False):
    """
    Plots all predicted vs real values of Z
    :param predicted: list of lists of predicted values
    :param real: list of lists of true values
    :param title: title of the plot
    :param no_kfold: number of kfold splits
    :param batch_no: batch size used
    :param no_epochs: number of epochs used
    :param split: if True, plots each k-fold separately
    :return: None
    """
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    fig.set_figheight(7)
    fig.set_figwidth(10)
    rmse_metric = RootMeanSquaredError()
    overall_score = []
    total_rmse = []
    if not isinstance(predicted, list):
        ax.plot(real, predicted, 'k.', alpha=0.5)
        ax.plot([real.min(), real.max()], [real.min(), real.max()], 'r--')
        overall_score = r2_score(real, predicted)
        rmse_metric.update_state(y_true=real, y_pred=predicted)
        total_rmse.append(rmse_metric.result().numpy())
    else:
        if split:
            marker_col = '.'
        else:
            marker_col = 'k.'

        for i, (r, p) in enumerate(zip(real, predicted)):
            ax.plot(r, p, marker_col, alpha=0.1, label=f'Fold {i+1}')
            overall_score.append(r2_score(r, p))
            rmse_metric.update_state(y_true=r, y_pred=p)
            total_rmse.append(rmse_metric.result().numpy())

        if split:
            ax.legend()

        real_all = [inner for outer in real for inner in outer]
        ax.plot([min(real_all), max(real_all)], [min(real_all), max(real_all)], 'r--')

    overall_score = np.mean(overall_score)
    overall_rmse = np.mean(total_rmse)
    fig_text = f"R^2: {overall_score:.3f}\n" \
               f"RMSE: {overall_rmse:.3f}"
    fig.text(0.14, 0.81, fig_text, fontsize=12)

    more_info = f"Epochs = {no_epochs}\n" \
               f"Batch Size = {batch_no}\n" \
               f"K-Folds = {no_kfold}"
    fig.text(0.74, 0.13, more_info, fontsize=12)

    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'Comparing Predicted and Measured Z-Axis Displacement')
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
    df_norm = (df - df.min()) / (df.max() - df.min())
    return df_norm

def residuals_plot(predicted, real):
    plt.style.use('ggplot')
    fig, ax = plt.subplots()

    if not isinstance(predicted, list):
        residual = predicted - real
        sns.residplot(x = predicted, y = residual, data = None, scatter_kws={'color': 'k',
       'alpha': 0.1}, ax=ax)
    else:
        for i, (r, p) in enumerate(zip(real, predicted)):
            residual = p - r
            sns.residplot(x = p, y = residual, data = None, scatter_kws={'color': 'k',
           'alpha': 0.1}, ax=ax)

    ax.set_xlabel('Predicted Value')
    ax.set_ylabel('Residuals')
    plt.show()

