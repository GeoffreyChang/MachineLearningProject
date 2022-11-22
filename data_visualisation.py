import matplotlib.pyplot as plt
import seaborn as sns
from helper_functions import *
from scipy import stats
import numpy as np

def data_visual(df, title, save_fig):
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

def boxplot_visual(df, title, save_fig):
    """
        boxplot helps us find any outliers, if available
    """
    features, target = get_features_and_target(df)

    fig, axes = plt.subplots(1,2,figsize=(10,5))
    fig.tight_layout(pad=4)
    bp = features.boxplot(ax=axes[0])
    bp.set_title(f'Boxplots for {title}', fontsize=15)
    bp.set_xlabel("Features")
    bp.set_ylabel("Temperature")

    bp = target.plot.box(ax=axes[1])
    bp.set_xlabel("Target")
    bp.set_ylabel("Z")

    if save_fig:
        print(f"boxplot_{title} saved")
        plt.savefig(f'graphs/{title}_boxplot')
    else:
        plt.show()
    plt.close()

def heatmap(df, title, save_fig):
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
    individual = True

    if individual:
        files = read_all_files()
        for k, data in enumerate(files):
            tt = f"dataset_{k + 1}"
            print(f"Best Features for {tt}: {find_best_features(data)}")
            # data_visual(data, tt, save_plots)
            boxplot_visual(data, tt, save_plots)
            # heatmap(data, tt, save_plots)
    else:
        data = entire_dataset()
        tt = "All Datasets Combined"
        print(f"Best Features for {tt}: {find_best_features(data)}")
        heatmap(data, tt, save_plots)




