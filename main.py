from helper_functions import *
import pandas as pd
import timeit
test = False

if test:
    # Import necessary libraries
    import numpy as np
    from keras.layers import LSTM
    from keras.models import Sequential
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import GridSearchCV


    # Create a function that returns an LSTM model with different hyperparameters
    def create_model(units=32, batch_size=32, epochs=10):
        model = Sequential()
        model.add(LSTM(units=units))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model


    # Define the hyperparameters to tune
    param_grid = {
        'units': [32, 64, 128],
        'batch_size': [32, 64, 128],
        'epochs': [10, 20, 30]
    }

    # Create the LSTM model using KerasClassifier
    model = KerasClassifier(build_fn=create_model, verbose=0)

    # Use scikit-learn's GridSearchCV to tune the hyperparameters
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)

    # Fit the model to the training data
    grid_result = grid.fit(X_train, y_train)

    # Print the best hyperparameters
    print(f"Best units: {grid_result.best_params_['units']}")
    print(f"Best batch_size: {grid_result.best_params_['batch_size']}")
    print(f"Best epochs: {grid_result.best_params_['epochs']}")

else:
    import numpy as np

    # Load the time series data
    data = read_all_files(1)
    data = data.drop(["S"], axis=1)

    # Create a new array to store the supervised data
    supervised_data = np.empty((data.shape[0], data.shape[1]))

    # Set the first column of the supervised data to the time values from the original data
    supervised_data[:, 0] = data["TIME"]

    # Set the remaining columns of the supervised data to the shifted values of the original data
    for col in range(1, data.shape[1]-1):
        supervised_data[col:, col] = data.iloc[:-col, col]

    # Set the last column of the supervised data to the target variable from the original data
    supervised_data[:, -1] = data["Z"]

    # Drop the first row of the supervised data, which contains NAN values
    supervised_data = supervised_data[1:, :]

    # Save the supervised data to a new file
    # np.save('supervised_data.npy', supervised_data)



