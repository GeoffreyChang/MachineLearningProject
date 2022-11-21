import os
import glob
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
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    ax.plot(real, predicted, 'k.')
    ax.plot([real.min(), real.max()], [real.min(), real.max()], 'r--')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('R2: {:.3f}'.format(r2_score(real, predicted)))
    plt.show()
