import numpy as np
from sklearn.metrics import r2_score

from helper_functions import *
from tensorflow import keras
if __name__ == "__main__":
    # files = read_all_files()
    df = read_file_no(1)
    features, target = get_features_and_target(df)
    normalizer = keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(features))


    # Linear Regression Model - Use single varialbe "T1" to predict Z axis
    t1 = np.array(features['T1'])

    t1_normalizer = keras.layers.Normalization(input_shape=[1, ], axis=None)
    t1_normalizer.adapt(t1)

    t1_model = keras.Sequential([
        t1_normalizer,
        keras.layers.Dense(units=1)
    ])

    t1_model.summary()

    t1_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.1),
        loss='mse')

    history = t1_model.fit(
        features['T1'],
        target,
        epochs=100,
        verbose=1,
        validation_split=0.2)

    i = 2
    while True:
        df2 = read_file_no(i)
        features2, target2 = get_features_and_target(df2)
        y_hat = t1_model.predict(features2["T1"])
        y_test = target2

        score = r2_score(y_test, y_hat)
        print("R square: %.3f" % score)
        z_plot_comparison(y_hat, y_test)

        t1_model.save('partly_trained.tf')
        del t1_model

        t1_model = keras.models.load_model('partly_trained.tf')

        t1_model.fit(
            features2['T1'],
            target2,
            epochs=100,
            verbose=1,
            validation_split=0.2)
        if i == 14:
            break
        i += 1
