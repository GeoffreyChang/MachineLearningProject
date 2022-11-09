import matplotlib.pyplot as plt
import os
import glob
import pandas as pd


path = os.getcwd()
path = os.path.join(path, "dataset/")
excel_files = glob.glob(os.path.join(path, "*.xlsx"))


if __name__ == "__main__":
    cols = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12']
    for f in excel_files:
        df = pd.read_excel(f)
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