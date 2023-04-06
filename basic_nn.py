import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Conv1D, Flatten, GRU, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

import absl.logging
from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error as mae, mean_absolute_percentage_error as mape
from statsmodels.tsa.seasonal import seasonal_decompose

import os
import time
import pandas as pd
import numpy as np
import keras_tuner as kt
    

def bnn_get_metrics(predictions, actual):

    MSE = mse(actual, predictions, squared=True)
    MAE = mae(actual, predictions)
    MAPE = mape(actual, predictions)
    RMSE = mse(actual, predictions, squared=False)
    R2 = r2_score(actual, predictions)

    metrics = {'RMSE': RMSE, 'R2': R2, 'MSE': MSE, 'MAE': MAE, 'MAPE': MAPE}
    return metrics


def bnn_make_csvs(csv_directory, predictions, y_test, pred_dates_test, set_name, future, time):

    metric_outputs = bnn_get_metrics(predictions, y_test)

    if not os.path.exists(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv"):
        performances = pd.DataFrame({"Date":pred_dates_test, "Actual": y_test, "Basic_nn": predictions})
        performances.to_csv(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv", index=False)
    else:
        performances = pd.read_csv(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv")
        performances['Basic_nn'] = predictions
        performances.to_csv(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv", index=False)

    if not os.path.exists(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv"):
        new_row = {'Model': ["Basic_nn"], 'RMSE': [metric_outputs.get("RMSE")], 'R2': [metric_outputs.get("R2")], 
                    'MSE': [metric_outputs.get("MSE")], 'MAE': [metric_outputs.get("MAE")], 
                    'MAPE': [metric_outputs.get("MAPE")], "TIME": [time]}

        metrics = pd.DataFrame(new_row)
        metrics.to_csv(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv", index=False)
    else:

        metrics = pd.read_csv(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv")

        if 'Basic_nn' in metrics['Model'].values:
            metrics.loc[metrics['Model'] == 'Basic_nn', 'RMSE'] = metric_outputs.get("RMSE")
            metrics.loc[metrics['Model'] == 'Basic_nn', 'R2'] = metric_outputs.get("R2")
            metrics.loc[metrics['Model'] == 'Basic_nn', 'MSE'] = metric_outputs.get("MSE")
            metrics.loc[metrics['Model'] == 'Basic_nn', 'MAE'] = metric_outputs.get("MAE")
            metrics.loc[metrics['Model'] == 'Basic_nn', 'MAPE'] = metric_outputs.get("MAPE")
            metrics.loc[metrics['Model'] == 'Basic_nn', 'TIME'] = time
        else:
            new_row = {'Model': "Basic_nn", 'RMSE': metric_outputs.get("RMSE"), 'R2': metric_outputs.get("R2"), 
                        'MSE': metric_outputs.get("MSE"), 'MAE': metric_outputs.get("MAE"), 
                        'MAPE': metric_outputs.get("MAPE")}

            metrics = metrics.append(new_row, ignore_index=True)
        metrics.to_csv(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv", index=False)


def bnn_kt_model(hp):

    hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
    hp_learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    hp_reg = hp.Float("reg", min_value=1e-4, max_value=1e-2, sampling="log")
    hp_dropout = hp.Float("dropout", min_value=1e-3, max_value=0.5, sampling="linear")
    hp_neuron_pct = hp.Float('NeuronPct', min_value=1e-3, max_value=1.0, sampling='linear')
    hp_neuron_shrink = hp.Float('NeuronShrink', min_value=1e-3, max_value=1.0, sampling='linear')
    
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
    

def bnn_train_model(future, batch_size, epochs,
                model_directory, set_name, X_train, y_train):
    
    start_time = time.time()

    tuner = kt.Hyperband(bnn_kt_model, objective='mean_absolute_percentage_error', max_epochs=epochs, factor=3, 
                        directory=model_directory + "/" + set_name + "_kt_dir", project_name='kt_model_' + str(future), 
                        overwrite=True)

    monitor = EarlyStopping(monitor='mean_absolute_percentage_error', min_delta=1, patience=5, verbose=0, mode='auto', 
                    restore_best_weights=True)

    tuner.search(X_train, y_train, verbose=0, epochs=epochs, validation_split=0.2, batch_size=batch_size,
                callbacks=[monitor])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X_train, y_train, verbose=1, epochs=epochs, callbacks=[monitor],
                    batch_size=batch_size, validation_split=0.2)
    model.save(model_directory + "/" + set_name + "_basic_nn_" + str(future))

    end_time = time.time()
    total_time = start_time - end_time

    return total_time


def bnn_predict(future, set_name, pred_dates_test, X_test, y_test, y_scaler, time):

    folder_path = os.getcwd()
    model_directory = folder_path + r"\models"
    csv_directory = folder_path + r"\csvs"

    model = load_model(model_directory + "/" + set_name + "_basic_nn_" + str(future))
    predictions = model.predict(X_test)
    print(predictions.shape)
    predictions = y_scaler.inverse_transform(predictions).reshape(-1)
    y_test = y_scaler.inverse_transform(y_test).reshape(-1)

    bnn_make_csvs(csv_directory, predictions, y_test, pred_dates_test, set_name, future, time)
    print(f"Finished running basic prediction on future window {0}", future)


def bnn_evaluate(future, set_name, X_train, y_train, epochs, batch_size):

    folder_path = os.getcwd()
    model_directory = folder_path + r"\models"

    absl.logging.set_verbosity(absl.logging.ERROR)
    tf.compat.v1.logging.set_verbosity(30)

    bnn_train_model(future, batch_size, epochs,
            model_directory, set_name, X_train, y_train)
    
    print("Finished evaluating basic nn for future {0}".format(future))
    
