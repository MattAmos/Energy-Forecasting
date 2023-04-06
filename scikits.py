import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Conv1D, Flatten, GRU, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

import absl.logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error as mae, mean_absolute_percentage_error as mape
from statsmodels.tsa.seasonal import seasonal_decompose

import os
import pandas as pd
import numpy as np
import keras_tuner as kt


# Biggest thing is finding out what these directories should be called. And the hyperparameter tuning library too

class SciFeatureScaler(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):

        dry = MinMaxScaler(feature_range=(0,1))
        dew = MinMaxScaler(feature_range=(0,1))
        wet = MinMaxScaler(feature_range=(0,1))
        humid = MinMaxScaler(feature_range=(0,1))
        hour = MinMaxScaler(feature_range=(0,1))
        prevWeek = MinMaxScaler(feature_range=(0,1))
        prevDay = MinMaxScaler(feature_range=(0,1))
        prev24 = MinMaxScaler(feature_range=(0,1))

        X['DryBulb'] = dry.fit_transform(X[['DryBulb']])
        X['DewPnt'] = dew.fit_transform(X[['DewPnt']])
        X['WetBulb'] = wet.fit_transform(X[['WetBulb']])
        X['Humidity'] = humid.fit_transform(X[['Humidity']])
        X['Hour'] = hour.fit_transform(X[['Hour']])
        X['PrevWeekSameHour'] = prevWeek.fit_transform(X[['PrevWeekSameHour']])
        X['PrevDaySameHour'] = prevDay.fit_transform(X[['PrevDaySameHour']])
        X['Prev24HourAveLoad'] = prev24.fit_transform(X[['Prev24HourAveLoad']])

        return [dry, dew, wet, humid, hour, prevWeek, prevDay, prev24]
    
    
def sci_create_dataset(input, win_size):
    
    np_data = input.copy()

    X = []

    for i in range(len(np_data)-win_size):
        row = [r for r in np_data[i:i+win_size]]
        X.append(row)

    return np.array(X)


def sci_get_metrics(predictions, actual):

    MSE = mse(actual, predictions, squared=True)
    MAE = mae(actual, predictions)
    MAPE = mape(actual, predictions)
    RMSE = mse(actual, predictions, squared=False)
    R2 = r2_score(actual, predictions)

    metrics = {'RMSE': RMSE, 'R2': R2, 'MSE': MSE, 'MAE': MAE, 'MAPE': MAPE}
    return metrics
    

def sci_scaling(csv_directory, future, set_name):

    data = pd.read_csv(csv_directory + "/" + set_name + "_data_" + str(future) + ".csv").drop('Date', axis=1)
    outputs = pd.read_csv(csv_directory + "/" + set_name + "_outputs_" + str(future) + ".csv")

    pipe = Pipeline([('Scaler', sciFeatureScaler())])
    scalers = pipe.fit_transform(data)

    pred_dates = outputs['Date']

    y_scaler = MinMaxScaler(feature_range=(0,1))

    y_data = y_scaler.fit_transform(outputs[['SYSLoad']])

    X_frame = np.array(data)
    y_data = np.array(y_data)

    return X_frame, y_data, pred_dates, y_scaler, scalers


def sci_make_csvs(csv_directory, predictions, y_test, pred_dates_test, set_name, future):

    metric_outputs = sci_get_metrics(predictions, y_test)

    if not os.path.exists(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv"):
        performances = pd.DataFrame({"Date":pred_dates_test, "Actual": y_test, "Complex_nn": predictions})
        performances.to_csv(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv", index=False)
    else:
        performances = pd.read_csv(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv")
        performances['Complex_nn'] = predictions
        performances.to_csv(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv", index=False)

    if not os.path.exists(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv"):
        new_row = {'Model': ["Complex_nn"], 'RMSE': [metric_outputs.get("RMSE")], 'R2': [metric_outputs.get("R2")], 
                    'MSE': [metric_outputs.get("MSE")], 'MAE': [metric_outputs.get("MAE")], 
                    'MAPE': [metric_outputs.get("MAPE")]}

        metrics = pd.DataFrame(new_row)
        metrics.to_csv(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv", index=False)
    else:

        metrics = pd.read_csv(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv")

        if 'Complex_nn' in metrics['Model'].values:
            metrics.loc[metrics['Model'] == 'Complex_nn', 'RMSE'] = metric_outputs.get("RMSE")
            metrics.loc[metrics['Model'] == 'Complex_nn', 'R2'] = metric_outputs.get("R2")
            metrics.loc[metrics['Model'] == 'Complex_nn', 'MSE'] = metric_outputs.get("MSE")
            metrics.loc[metrics['Model'] == 'Complex_nn', 'MAE'] = metric_outputs.get("MAE")
            metrics.loc[metrics['Model'] == 'Complex_nn', 'MAPE'] = metric_outputs.get("MAPE")
        else:
            new_row = {'Model': "Complex_nn", 'RMSE': metric_outputs.get("RMSE"), 'R2': metric_outputs.get("R2"), 
                        'MSE': metric_outputs.get("MSE"), 'MAE': metric_outputs.get("MAE"), 
                        'MAPE': metric_outputs.get("MAPE")}

            metrics = metrics.append(new_row, ignore_index=True)
        metrics.to_csv(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv", index=False)
    

def sci_train_model(X_frame, y_data, window, future, batch_size, split, epochs, pred_dates, y_scaler,
                model_directory, csv_directory, set_name):

    length = X_frame.shape[0]

    pred_dates_test = pred_dates[int(length*split) + window:]

    y_train = y_data[window:int(length*split)]
    y_test = y_data[int(length*split) + window:]

    X_train = sci_create_dataset(X_frame[:int(length*split)], window)
    X_test = sci_create_dataset(X_frame[int(length*split):], window)

    # Need to do this entire second part for each of the models we are making: svm, rf and xgboost

    model.save(model_directory + "/" + set_name + "_Complex_nn_" + str(future))

def sci_predict(window, future, set_name):

    folder_path = os.getcwd()
    model_directory = folder_path + r"\models"
    csv_directory = folder_path + r"\csvs"

    X_frame, y_data, pred_dates, y_scaler, _ = sci_scaling(csv_directory, future, set_name)

    length = X_frame.shape[0]
    split = 0.7

    pred_dates_test = pred_dates[int(length*split) + window:]
    y_test = y_data[int(length*split) + window:]
    X_test = sci_create_dataset(X_frame[int(length*split):], window)

    model = load_model(model_directory + "/" + set_name + "_what should this be_" + str(future))
    predictions = model.predict(X_test)
    predictions = y_scaler.inverse_transform(predictions).reshape(-1)
    y_test = y_scaler.inverse_transform(y_test).reshape(-1)

    sci_make_csvs(csv_directory, predictions, y_test, pred_dates_test, set_name, future)
    print(f"Finished running sci prediction on future window {0}", future)


def sci_evaluate(window, future, set_name):

    folder_path = os.getcwd()
    model_directory = folder_path + r"\models"
    csv_directory = folder_path + r"\csvs"

    epochs = 200
    batch_size = 32
    split = 0.7

    absl.logging.set_verbosity(absl.logging.ERROR)

    try:

        X_frame, y_data, pred_dates, y_scaler, _ = sci_scaling(csv_directory, future, set_name)

        sci_train_model(X_frame, y_data, window, future, batch_size, split, epochs, pred_dates, y_scaler, 
                model_directory, csv_directory, set_name)
        
        print("Finished evaluating scikit models for future {0}".format(future))

    except FileNotFoundError:

        print("Files are not present in \"csvs\" directory.")
        print("Ensure they are before continuing")
    
