import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import seaborn as sns
from helper_functions import *


def data_visual(df, save_fig):
    """
        Basic visualisation of the given dataset
    """
    fg, ax = plt.subplots(figsize=(15, 10))
    for i in range(1, 15):
        temp = "T" + str(i)
        try:
            ax.plot(df["TIME"], df[temp], '-', label=temp)
        except KeyError:
            pass
    ax.set_title(f'Basic Visual of {title}', fontsize=26)
    plt.xlabel('Time', fontsize=20, labelpad=20)
    plt.ylabel('Temperature', fontsize=20, labelpad=20)

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
    plt.close()

def boxplot(df, save_fig):
    """
        boxplot helps us find any outliers, if available
    """
    df = df.drop(["TIME", "S", "Z"], axis=1)
    plt.figure()
    bp = df.boxplot()
    bp.set_title(f'Boxplot for {title}', fontsize=15)
    bp.set_xlabel("Variable")
    bp.set_ylabel("Temperature")

    if save_fig:
        print(f"boxplot_{title} saved")
        plt.savefig(f'graphs/{title}_boxplot')
    else:
        plt.show()
    plt.close()

def heatmap(df, save_fig):
    """
        Heatmap helps us find correlation between variables
    """
    plt.figure()
    sns.set(font_scale=0.6)
    df = df.drop(["TIME", "S"], axis=1)
    corr = df.corr()

    plt.title(f'Heatmap for {title}', fontsize=15)
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    if save_fig:
        print(f"heatmap_{title} saved")
        plt.savefig(f'graphs/{title}_heatmap')
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    # Plot style
    plt.style.use('ggplot')
    # to save as png
    save_plots = False
    # plot individual datasets or entire dataset
    individual = False

    if individual:
        files = read_all_files()
        for k, data in enumerate(files):
            title = f"dataset_{k + 1}"
            data_visual(data, save_plots)
            boxplot(data, save_plots)
            heatmap(data, save_plots)
    else:
        data = entire_dataset()
        title = "All Datasets combined"
        boxplot(data, save_plots)
        heatmap(data, save_plots)

    files = read_all_files()
    files[0].corrwith(files[1], axis=0)


