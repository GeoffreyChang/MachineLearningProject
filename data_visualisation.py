import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

path = os.getcwd()
path = os.path.join(path, "dataset/")
excel_files = glob.glob(os.path.join(path, "*.xlsx"))


def data_visual(df, save_fig):
    """
        Basic visualisation of the given dataset
    """
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

    if save_fig:
        print(f"{title} saved")
        plt.savefig(f'graphs/{title}')
    else:
        fg.show()

def boxplot(df, save_fig):
    """
        boxplot helps us find any outliers, if available
    """
    df.drop(columns=df.columns[[0, 1, -1]], axis=1, inplace=True)

    bp = df.boxplot()
    title = f.split("\\")[-1][26:-5]
    bp.set_title(f'{title}')
    bp.set_xlabel("Variable")
    bp.set_ylabel("Temperature")

    if save_fig:
        print(f"boxplot_{title} saved")
        plt.savefig(f'graphs/boxplot_{title}')
    else:
        plt.show()

if __name__ == "__main__":
    for f in excel_files:
        data = pd.read_excel(f)
        # Plot style
        plt.style.use('ggplot')

        # to save as png
        save_plots = False
        data_visual(data, save_plots)
        boxplot(data, save_plots)
