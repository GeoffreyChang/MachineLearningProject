import pandas as pd
import numpy as np
import os
import typing
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers
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

    print(f"route_distances shape={route_distances.shape}")
    print(f"speeds_array shape={speeds_array.shape}")