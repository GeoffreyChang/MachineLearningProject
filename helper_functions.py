import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def read_all_files():
    path = os.getcwd()
    path = os.path.join(path, "dataset/")
    excel_files = glob.glob(os.path.join(path, "*.xlsx"))
    main_df = []
    for f in excel_files:
        main_df.append(pd.read_excel(f))
    return main_df

def get_features_and_target(df):
    df.drop(columns=df.columns[[0, 1]], axis=1, inplace=True)
    df = df.dropna()
    features = df.copy()
    target = df.iloc[:, -1]
    features.drop(columns=features.columns[[-1]], axis=1, inplace=True)

    return features, target

def read_file_no(n):
    path = os.getcwd()
    path = os.path.join(path, "dataset/")
    excel_files = glob.glob(os.path.join(path, "*.xlsx"))
    df = pd.read_excel(excel_files[n-1])
    return df

def z_plot_comparison(predicted, real, r_val=0.0):
    plt.title("R^2 Value: %.3f" % r_val)
    plt.plot(predicted, 'r-', label='predicted')
    plt.plot(real, 'k-', label='real')
    plt.legend()
    plt.show()

def retrain_model(model, X, y):
    pass