import pandas as pd
import numpy as np
import os
import typing
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import timeseries_dataset_from_array


# https://keras.io/examples/timeseries/timeseries_traffic_forecasting/
if __name__ == "__main__":
    speeds_array = pd.read_csv('V_228.csv', header=None).to_numpy()
    route_distances = pd.read_csv('W_228.csv', header=None).to_numpy()

    sample_routes = [
        0,
        1,
        4,
        7,
        8,
        11,
        15,
        108,
        109,
        114,
        115,
        118,
        120,
        123,
        124,
        126,
        127,
        129,
        130,
        132,
        133,
        136,
        139,
        144,
        147,
        216,
    ]
    route_distances = route_distances[np.ix_(sample_routes, sample_routes)]
    speeds_array = speeds_array[:, sample_routes]

    plt.figure(figsize=(18, 6))
    plt.plot(speeds_array[:, [0, -1]])
    plt.legend(["route_0", "route_25"])
    # plt.show()

    train_size, val_size = 0.5, 0

    def preprocess(data_array: np.ndarray, train_size: float, val_size: float):
        """Splits data into train/val/test sets and normalizes the data.

        Args:
            data_array: ndarray of shape `(num_time_steps, num_routes)`
            train_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
                to include in the train split.
            val_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
                to include in the validation split.

        Returns:
            `train_array`, `val_array`, `test_array`
        """

        num_time_steps = data_array.shape[0]
        num_train, num_val = (
            int(num_time_steps * train_size),
            int(num_time_steps * val_size),
        )
        train_array = data_array[:num_train]
        mean, std = train_array.mean(axis=0), train_array.std(axis=0)

        train_array = (train_array - mean) / std
        val_array = (data_array[num_train: (num_train + num_val)] - mean) / std
        test_array = (data_array[(num_train + num_val):] - mean) / std

        return train_array, val_array, test_array

    train_array, val_array, test_array = preprocess(speeds_array, train_size, val_size)
    print(f"train set size: {train_array.shape}")
    print(f"validation set size: {val_array.shape}")
    print(f"test set size: {test_array.shape}")