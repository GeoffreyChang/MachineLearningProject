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
    plt.title(f'Boxplots for {title}', fontsize=15)
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

def plot_histogram(df):
    features, targets = get_features_and_target(df)

    for f in features:
        plt.figure()
        plt.title(f)
        sns.distplot(features[f], fit=stats.norm)
        plt.show()
        plt.close()
    # plt.hist(features)
    # plt.show()

def plot_data_3_by_3(n1, n2, n3, save_fig=False):
    df1 = read_all_files(n1)
    df2 = read_all_files(n2)
    df3 = read_all_files(n3)

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    fig.tight_layout(pad=2.65)
    # df1
    axs[0, 0].plot(df1["TIME"], df1["S"], 'k-')
    axs[0, 0].set_title("Spindle Speed")

    axs[1, 0].plot(df1["TIME"], df1.iloc[:, 2:12], '-')
    axs[1, 0].set_title("Temperature")

    axs[2, 0].plot(df1["TIME"], df1["Z"], 'k-')
    axs[2, 0].set_title("Thermal Displacement")

    axs[2, 0].set_xlabel(f"Time (min) \n [Dataset {n1}]")

    # df2
    axs[0, 1].plot(df2["TIME"], df2["S"], 'k-')
    axs[0, 1].set_title("Spindle Speed")

    axs[1, 1].plot(df2["TIME"], df2.iloc[:, 2:12], '-')
    axs[1, 1].set_title("Temperature")

    axs[2, 1].plot(df2["TIME"], df2["Z"], 'k-')
    axs[2, 1].set_title("Thermal Displacement")

    axs[2, 1].set_xlabel(f"Time (min) \n [Dataset {n2}]")

    # df3
    axs[0, 2].plot(df3["TIME"], df3["S"], 'k-')
    axs[0, 2].set_title("Spindle Speed")

    axs[1, 2].plot(df3["TIME"], df3.iloc[:, 2:12], '-')
    axs[1, 2].set_title("Temperature")

    axs[2, 2].plot(df3["TIME"], df3["Z"], 'k-')
    axs[2, 2].set_title("Thermal Displacement")

    axs[2, 2].set_xlabel(f"Time (min) \n [Dataset {n3}]")
    # for ax in axs.flat:
    #     ax.set(xlabel='Time (min)')

    for ax in axs.flat:
        ax.label_outer()


    plt.show()

if __name__ == "__main__":
    plot_data_3_by_3(13, 14, 1)
    # # Plot style
    # plt.style.use('ggplot')
    # # to save as png
    # save_plots = False
    # # plot individual datasets or entire dataset
    # individual = False
    # # plot_histogram(read_file_no(2))
    # if individual:
    #     files = read_all_files()
    #     for k, data in enumerate(files):
    #         tt = f"dataset_{k + 1}"
    #         print(f"Best Features for {tt}: {find_best_features(data)}")
    #         # data_visual(data, tt, save_plots)
    #         boxplot_visual(data, tt, save_plots)
    #         # heatmap(data, tt, save_plots)
    # else:
    #     data = read_all_files()
    #     data.pop(9) # remove the 10th file due to missing data
    #     data.pop(5) # remove the 6th file due to extremely high values
    #     data = pd.concat(data)
    #     tt = "All Datasets Combined except dataset6 and dataset10"
    #     boxplot_visual(data, tt, save_plots)
    #     # print(f"Best Features for {tt}: {find_best_features(data)}")
    #     # heatmap(data, tt, save_plots)






