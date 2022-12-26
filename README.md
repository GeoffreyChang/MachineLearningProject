# Thermal Displacement Compensation in CNC Manufacturing
This repository contains the Python code and data used in a study on the use of machine learning algorithms for real-time compensation of thermal displacement in CNC manufacturing processes. The goal of the study was to investigate the potential of machine learning to improve the precision and efficiency of CNC manufacturing processes.

### Further Background Information
The goal of the study was to investigate the potential of machine learning to improve the precision and efficiency of CNC manufacturing processes based on temperature data. I was provided with 13 datasets, each containing 15 variables including time, RPM, and various temperature points (T1-T12). I applied a range of machine learning models to the datasets, and found that some models performed better than others. In the report, I will discuss the different models that I applied and their performance on the datasets.


### Content
The repository contains the following files and directories:

`dataset/`: A folder containing multiple datasets from a thermal elongation machine, used to train and test the machine learning models.<br>
`graphs/`: A folder containing visualizations of the data.<br>
`GRU_model.py`: A Python script that implements a gated recurrent unit (GRU) model for thermal displacement compensation.<br>
`GRU_window_sliding_method.py`: A Python script that implements a window sliding method using a GRU model for thermal displacement compensation.<br>
`LSTM_model.py`: A Python script that implements a long short-term memory (LSTM) model for thermal displacement compensation.<br>
`LSTM_window_sliding_method.py` A Python script that implements a window sliding method using an LSTM model for thermal displacement compensation.<br>
`data_visualisation.py` A Python script that generates visualizations of the data.<br>
`dnn_regression.py` A Python script that implements a deep neural network (DNN) model for thermal displacement compensation using regression.<br>
`helper_functions.py` A Python script containing helper functions used by the other scripts.<br>
`model_comparison.py` A Python script that compares the performance of different machine learning models on the datasets.<br>
`ridge_regression.py` A Python script that implements a ridge regression model for thermal displacement compensation.<br>
`rnn.py` A Python script that implements a basic recurrent neural network (RNN) model for thermal displacement compensation.<br>
`svr_model.py` A Python script that implements a support vector regression (SVR) model for thermal displacement compensation.<br>

### The code in this repository requires the following Python packages:

* `numpy`
* `pandas`
* `scikit-learn`
* `tensorflow`
  * `keras`
* `matplotlib`


### Usage
To run the compensation system, use the following command:

### Contact
For questions or feedback, please contact Geoffrey Chang at geoffreychang.nz@gmail.com.
