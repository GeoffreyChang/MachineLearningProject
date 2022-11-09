import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

path = os.getcwd()
path = os.path.join(path, "dataset/")
excel_files = glob.glob(os.path.join(path, "*.xlsx"))


def data_visual(df):
    fg, ax = plt.subplots(figsize=(15, 10))
    for i in range(1, 20):
        temp = "T" + str(i)
        try:
            ax.plot(df.iloc[:, 0], df[temp], label=temp)
        except KeyError:
            pass
    title = f.split("\\")[-1][26:-5]
    plt.title(f'{title}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature')

    ax2 = ax.twinx()
    ax2.plot(df.iloc[:, 0], df["Z"], 'k--', label="Z")
    ax2.set_ylabel('Z Axis')
    handles, labels = [(a + b) for a, b in zip(ax.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 0.5))

    fg.show()
    save_fig = False
    if save_fig:
        plt.savefig(f'graphs/{title}')



if __name__ == "__main__":
    for i, f in enumerate(excel_files):
        data = pd.read_excel(f)
        # data_visual(data)