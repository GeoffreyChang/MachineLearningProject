import pandas as pd
import seaborn as sns
import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

path = os.getcwd()
path = os.path.join(path, "Dataset/")
excel_files = glob.glob(os.path.join(path, "*.xlsx"))

if __name__ == "__main__":
    main_df = []
    for f in excel_files:
        df = pd.read_excel(f)


        normalized_df=(df-df.min())/(df.max()-df.min())



        break


