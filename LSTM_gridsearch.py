import LSTM_window_sliding_method
import GRU_window_sliding_method
import LSTM_model
import GRU_model
import dnn_regression
import main

if __name__ == "__main__":

    best_params = None
    best_score = float('inf')

    param_grid = {
        'dense_units': [8, 16, 32, 64, 128],
        'num_dense': [1, 2, 3, 4]
    }

    no_kfold = 10
    no_window = 3
    # arr_epoc = list(range(130, 210, 10))  # [10, 20, 30, ..., 200]
    # batch = 16

    for ee in param_grid['dense_units']:
        for rr in param_grid['num_dense']:

            rmse_score = LSTM_window_sliding_method.main(epoch=100, batch_no=32, kfold_splits=no_kfold, window_size=no_window)
            if rmse_score < best_score:
                best_params = {'units in each layer': ee, 'number of layers': rr}
                best_score = rmse_score

    # print the best hyperparameters and evaluation score
    print(f'Best params: {best_params}')
    print(f'Best score: {best_score}')