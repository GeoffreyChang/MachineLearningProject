import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def read_all_files():
    path = os.getcwd()
    path = os.path.join(path, "dataset/")
    excel_files = glob.glob(os.path.join(path, "*.xlsx"))
    main_df = []
    for f in excel_files:
        df = pd.read_excel(f)
        df.rename(columns={df.columns[0]: 'TIME', df.columns[1]: 'S'}, inplace=True)
        main_df.append(df)
    return main_df
def entire_dataset():
    return pd.concat(read_all_files())

def get_features_and_target(df):
    df = df.drop(["TIME", "S"], axis=1)
    df = df.dropna()
    features = df.copy()
    target = df["Z"]
    features.drop(["Z"], axis=1, inplace=True)
    return features, target

def read_file_no(n):
    path = os.getcwd()
    path = os.path.join(path, "dataset/")
    excel_files = glob.glob(os.path.join(path, "*.xlsx"))
    df = pd.read_excel(excel_files[n-1])
    df.rename(columns={df.columns[0]: 'TIME', df.columns[1]: 'S'}, inplace=True)
    return df

def z_plot_comparison(predicted, real):
    """
    Plots predicted vs real values of Z

    param predicted: list of predicted values
    param real: list of true values
    return: None
    """
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    ax.plot(real, predicted, 'k.')
    ax.plot([real.min(), real.max()], [real.min(), real.max()], 'r--')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('R2: {:.3f}'.format(r2_score(real, predicted)))
    plt.show()

def z_plot_all(predicted, real):
    """
    Plots all predicted vs real values of Z
    :param predicted: list of lists of predicted values
    :param real: list of lists of true values
    :return: None
    """
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    ax.plot([real[0].min(), real[0].max()], [real[0].min(), real[0].max()], 'r--')
    overall_score = []
    for i, (r, p) in enumerate(zip(real, predicted)):
        ax.plot(r, p, '.', alpha=0.5, label=f'Fold {i+1}')
        overall_score.append(r2_score(r, p))
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.legend()
    ax.set_title('R2: {:.3f}'.format(np.mean(overall_score)))
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