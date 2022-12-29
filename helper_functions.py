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
    Reads in all CSV files in the "dataset" directory and returns a list of pandas DataFrames.

    If the optional parameter n is specified, only the n-th CSV file will be read and returned as a single DataFrame.

    Args:
        n: Optional file number to read (if specified, only this file will be read and returned).

    Returns:
        list of pandas DataFrames or a single DataFrame (if n is specified).
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
    """
    Returns a single pandas DataFrame containing all the data from the CSV files in the "dataset" directory.

    Returns:
        pandas DataFrame: A DataFrame containing all the data from the CSV files.
    """
    return pd.concat(read_all_files())

def get_features_and_target(df):
    try:
        df = df.drop(["TIME", "S"], axis=1)
    except KeyError:
        pass
    features, target = df.iloc[:, :-1], df["Z"]
    return features, target

def z_plot(predicted, real, no_epochs=0, batch_no=0, no_kfold=0, split=False, save_fig=""):
    """
    Plots the predicted and real values of the Z-axis displacement.

    Args:
        predicted (panda series or list of panda series): The predicted Z-axis displacement values. If a list is
            provided, each element of the list represents the predicted values for each fold of a cross-validation procedure.
        real (panda series or list of panda series): The real Z-axis displacement values. If a list is provided, each
            element of the list represents the real values for each fold of a cross-validation procedure.
        no_epochs (int, optional): The number of epochs used in the model training.
        batch_no (int, optional): The batch size used in the model training.
        no_kfold (int, optional): The number of folds used in the cross-validation procedure.
        split (bool, optional): Whether to show separate plots for each fold of the cross-validation procedure.
            Default is False.
        save_fig (str, optional): The file path to save the plot. If not provided, the plot will be displayed on the
            screen.

    Returns:
        None
    """
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    fig.set_figheight(7)
    fig.set_figwidth(10)

    overall_score = []
    total_rmse = []
    ll = False
    try:
        predicted[0][0]
    except IndexError:
        ll = True

    if ll:
        rmse_metric = RootMeanSquaredError()
        ax.plot(real, predicted, 'k.', alpha=0.5)
        ax.plot([min(real), max(real)], [min(real), max(real)], 'r--')
        overall_score = r2_score(real, predicted)
        rmse_metric.update_state(y_true=real, y_pred=predicted)
        total_rmse.append(rmse_metric.result().numpy())
    else:
        if split:
            marker_col = '.'
        else:
            marker_col = 'k.'

        for i, (r, p) in enumerate(zip(real, predicted)):
            rmse_metric = RootMeanSquaredError()
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
    if save_fig != "":
        if not os.path.isdir(f"graphs/z_plots/{save_fig}"):
            os.makedirs(f"graphs/z_plots/{save_fig}")
        plt.savefig("graphs/z_plots/" + save_fig + f"/epochs{no_epochs}_batch{batch_no}_kfolds{no_kfold}.png")
        plt.close()
    else:
        plt.show()
        plt.close()

def find_best_features(df):
    """
    Finds the most highly correlated features with the Z-axis displacement in the input DataFrame.

    Args:
        df (pandas DataFrame): The DataFrame containing the features and the Z-axis displacement.

    Returns:
        list: A list of strings of the most highly correlated features with the Z-axis displacement.
    """
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
    """
    Normalizes the values in the input DataFrame.

    Args:
        df (pandas DataFrame): The DataFrame to be normalized.

    Returns:
        pandas DataFrame: The normalized DataFrame.
    """
    df_norm = (df - df.min()) / (df.max() - df.min())
    return df_norm

def residuals_plot(predicted, real):
    """
    Plots the residuals (prediction errors) for the predicted and real values.

    Args:
        predicted (float or list): The predicted values. If a list is provided, each element of the list represents the
            predicted values for each fold of a cross-validation procedure.
        real (float or list): The real values. If a list is provided, each element of the list represents the real values
            for each fold of a cross-validation procedure.

    Returns:
        None
    """
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

