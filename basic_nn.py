import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Conv1D, Flatten, GRU, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

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




class FeatureScaler(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):

        if standard_scaling:
            dry = StandardScaler()
            dew = StandardScaler()
            wet = StandardScaler()
            humid = StandardScaler()
            hour = StandardScaler()
            prevWeek = StandardScaler()
            prevDay = StandardScaler()
            prev24 = StandardScaler()

        else:
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
    
    
def create_dataset(input, win_size):
    
    np_data = input.copy()

    X = []

    for i in range(len(np_data)-win_size):
        row = [r for r in np_data[i:i+win_size]]
        X.append(row)

    return np.array(X)


def get_metrics(predictions, actual):

    MSE = mse(actual, predictions, squared=True)
    MAE = mae(actual, predictions)
    MAPE = mape(actual, predictions)
    RMSE = mse(actual, predictions, squared=False)
    R2 = r2_score(actual, predictions)

    metrics = {'RMSE': RMSE, 'R2': R2, 'MSE': MSE, 'MAE': MAE, 'MAPE': MAPE}
    return metrics
    

def scaling(csv_directory, future):

    data = pd.read_csv(csv_directory + r"\data_" + str(future) + ".csv").drop('Date', axis=1)
    outputs = pd.read_csv(csv_directory + r"\outputs_" + str(future) + ".csv")

    pipe = Pipeline([('Scaler', FeatureScaler())])
    scalers = pipe.fit_transform(data)

    pred_dates = outputs['Date']
    actual = outputs['SYSLoad']

    if standard_scaling:
        y_scaler = StandardScaler()

    else:
        y_scaler = MinMaxScaler(feature_range=(0,1))

    y_data = y_scaler.fit_transform(outputs[['SYSLoad']])

    X_frame = np.array(data)
    y_data = np.array(y_data)

    return X_frame, y_data, pred_dates, y_scaler, scalers


def make_csvs(csv_directory, predictions, y_test, pred_dates_test):

    metric_outputs = get_metrics(predictions, y_test)

    if not os.path.exists(csv_directory + r"\performances_" + str(future) + ".csv"):
        performances = pd.DataFrame({"Date":pred_dates_test, "Actual": y_test, "Basic_nn": predictions})
        performances.to_csv(csv_directory + r"\performances_" + str(future) + ".csv", index=False)
    else:
        performances = pd.read_csv(csv_directory + r"\performances_" + str(future) + ".csv")
        performances['Basic_nn'] = predictions
        performances.to_csv(csv_directory + r"\performances_" + str(future) + ".csv", index=False)

    if not os.path.exists(csv_directory + r"\metrics_" + str(future) + ".csv"):
        new_row = {'Model': ["Basic_nn"], 'RMSE': [metric_outputs.get("RMSE")], 'R2': [metric_outputs.get("R2")], 
                    'MSE': [metric_outputs.get("MSE")], 'MAE': [metric_outputs.get("MAE")], 
                    'MAPE': [metric_outputs.get("MAPE")]}

        metrics = pd.DataFrame(new_row)
        metrics.to_csv(csv_directory + r"\metrics_" + str(future) + ".csv", index=False)
    else:

        metrics = pd.read_csv(csv_directory + r"\metrics_" + str(future) + ".csv")

        if 'Basic_nn' in metrics['Model'].values:
            metrics.loc[metrics['Model'] == 'Basic_nn', 'RMSE'] = metric_outputs.get("RMSE")
            metrics.loc[metrics['Model'] == 'Basic_nn', 'R2'] = metric_outputs.get("R2")
            metrics.loc[metrics['Model'] == 'Basic_nn', 'MSE'] = metric_outputs.get("MSE")
            metrics.loc[metrics['Model'] == 'Basic_nn', 'MAE'] = metric_outputs.get("MAE")
            metrics.loc[metrics['Model'] == 'Basic_nn', 'MAPE'] = metric_outputs.get("MAPE")
        else:
            new_row = {'Model': "Basic_nn", 'RMSE': metric_outputs.get("RMSE"), 'R2': metric_outputs.get("R2"), 
                        'MSE': metric_outputs.get("MSE"), 'MAE': metric_outputs.get("MAE"), 
                        'MAPE': metric_outputs.get("MAPE")}

            metrics = metrics.append(new_row, ignore_index=True)
        metrics.to_csv(csv_directory + r"\metrics_" + str(future) + ".csv", index=False)


def kt_model(hp):

    hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
    hp_learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    hp_reg = hp.Float("reg", min_value=1e-4, max_value=1e-2, sampling="log")
    hp_dropout = hp.Float("dropout", min_value=1e-3, max_value=0.5, sampling="linear")
    hp_neuron_pct = hp.Float('NeuronPct', min_value=1e-3, max_value=1.0, sampling='linear')
    hp_neuron_shrink = hp.Float('NeuronShrink', min_value=1e-3, max_value=1.0, sampling='linear')
    
    hp_l_layer_1 = hp.Int('l_layer_1', min_value=1, max_value=100, step=10)
    hp_max_neurons = hp.Int('neurons', min_value=10, max_value=200, step=10)

    neuron_count = int(hp_neuron_pct * hp_max_neurons)
    layers = 0

    model = Sequential()

    while neuron_count > 5 and layers < 5:

        model.add(Dense(units=neuron_count, activation=hp_activation))
        model.add(Dropout(hp_dropout))
        layers += 1
        neuron_count = int(neuron_count * hp_neuron_shrink)

    model.add(Dense(1, 'linear'))

    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=hp_learning_rate), 
                metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'])

    return model
    

def train_model(X_frame, y_data, window, future, batch_size, split, epochs, pred_dates, y_scaler,
                model_directory, csv_directory):

    length = X_frame.shape[0]

    pred_dates_test = pred_dates[int(length*split) + window:]

    y_train = y_data[window:int(length*split)]
    y_test = y_data[int(length*split):]

    X_train = create_dataset(X_frame[:int(length*split)], window)
    X_test = create_dataset(X_frame[int(length*split):], window)

    tuner = kt.Hyperband(kt_model, objective='mean_squared_error', max_epochs=epochs, factor=3, 
                        directory=model_directory + r'\kt_dir', project_name='kt_model_' + str(future), 
                        overwrite=True)

    monitor = EarlyStopping(monitor='loss', min_delta=1e-5, patience=5, verbose=0, mode='auto', 
                    restore_best_weights=True)

    tuner.search(X_train, y_train, verbose=1, epochs=epochs, validation_split=0.2, batch_size=batch_size,
                callbacks=[monitor])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X_train, y_train, verbose=1, epochs=epochs, callbacks=[monitor],
                    batch_size=batch_size, validation_split=0.2)
    model.save(model_directory + r'\basic_nn_' + str(future))

    predictions = model.predict(X_test)
    predictions = y_scaler.inverse_transform(predictions).reshape(-1)
    y_test = y_scaler.inverse_transform(y_test).reshape(-1)

    make_csvs(csv_directory, predictions, y_test, pred_dates_test)
    print(f"Finished running simulation on future window {0}", future)


if __name__=='__main__':

    folder_path = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.basename(folder_path)
    model_directory = folder_path + r"\models"
    csv_directory = folder_path + r"\csvs"

    standard_scaling = 1
    epochs = 200
    batch_size = 32
    future = 0
    window = 3
    split = 0.7

    absl.logging.set_verbosity(absl.logging.ERROR)
    tf.compat.v1.logging.set_verbosity(30)

    try:

        X_frame, y_data, pred_dates, y_scaler, scalers = scaling(csv_directory, future)

        train_model(X_frame, y_data, window, future, batch_size, split, epochs, pred_dates, y_scaler, 
                model_directory, csv_directory)

    except FileNotFoundError:

        print("Files are not present in \"csvs\" directory.")
        print("Ensure they are before continuing")
    
