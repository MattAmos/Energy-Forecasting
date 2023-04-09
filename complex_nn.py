import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Conv1D, Flatten, GRU, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

import absl.logging
from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error as mae, mean_absolute_percentage_error as mape
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.seasonal import seasonal_decompose

import os
import time
import pandas as pd
import numpy as np
import keras_tuner as kt
import matplotlib.pyplot as plt
    

def cnn_get_metrics(predictions, actual, cv):

    MSE = mse(actual, predictions, squared=True)
    MAE = mae(actual, predictions)
    MAPE = mape(actual, predictions)
    RMSE = mse(actual, predictions, squared=False)
    R2 = r2_score(actual, predictions)
    if cv:
        metrics = {'CNN_RMSE': RMSE, 'CNN_R2': R2, 'CNN_MSE': MSE, 'CNN_MAE': MAE, 'CNN_MAPE': MAPE}
    else:
        metrics = {'RMSE': RMSE, 'R2': R2, 'MSE': MSE, 'MAE': MAE, 'MAPE': MAPE}
    return metrics


def cnn_cross_val_metrics(total_metrics, set_name, future):

    csv_directory = os.getcwd() + "/csvs"
    df = pd.DataFrame(total_metrics)
    df.to_csv(csv_directory + "/" + set_name + "_cv_metrics_" + str(future) + ".csv", index=False)


def cnn_make_csvs(csv_directory, predictions, y_test, pred_dates_test, set_name, future, time):

    metric_outputs = cnn_get_metrics(predictions, y_test, 0)

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
                    'MAPE': [metric_outputs.get("MAPE")], "TIME": [time]}

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
            metrics.loc[metrics['Model'] == 'Complex_nn', 'TIME'] = time
        else:
            new_row = {'Model': "Complex_nn", 'RMSE': metric_outputs.get("RMSE"), 'R2': metric_outputs.get("R2"), 
                        'MSE': metric_outputs.get("MSE"), 'MAE': metric_outputs.get("MAE"), 
                        'MAPE': metric_outputs.get("MAPE")}

            metrics.loc[len(metrics)] = new_row
        metrics.to_csv(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv", index=False)


def cnn_kt_model(hp):

    X = np.load("X_train_3d.npy")

    hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
    hp_learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    hp_reg = hp.Float("reg", min_value=1e-4, max_value=1e-2, sampling="log")
    hp_dropout = hp.Float("dropout", min_value=1e-3, max_value=0.5, sampling="linear")
    hp_neuron_pct = hp.Float('NeuronPct', min_value=1e-3, max_value=1.0, sampling='linear')
    hp_neuron_shrink = hp.Float('NeuronShrink', min_value=1e-3, max_value=1.0, sampling='linear')
    
    hp_l_layer_1 = hp.Int('l_layer_1', min_value=1, max_value=100, step=10)
    hp_max_neurons = hp.Int('neurons', min_value=10, max_value=5000, step=10)

    neuron_count = int(hp_neuron_pct * hp_max_neurons)
    layers = 0

    model = Sequential()
    model.add(InputLayer((X.shape[1], X.shape[2])))
    model.add(LSTM(hp_l_layer_1, return_sequences=True, activity_regularizer=regularizers.l1(hp_reg)))
    model.add(Dropout(hp_dropout))
    model.add(Flatten())

    while neuron_count > 20 and layers < 20:

        model.add(Dense(units=neuron_count, activation=hp_activation))
        model.add(Dropout(hp_dropout))
        layers += 1
        neuron_count = int(neuron_count * hp_neuron_shrink)

    model.add(Dense(1, 'linear'))

    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=hp_learning_rate), 
                metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'])

    return model


def cnn_save_plots(history, graphs_directory, set_name, future):

    graph_names = {"Loss": "loss", "MAE": "mean_absolute_error", 
                   "MSE": "mean_squared_error", "MAPE": "mean_absolute_percentage_error"}
    
    for name, value in graph_names.items():
        graph_loc = graphs_directory + "/" + set_name + "_Complex_nn_" + str(future) + "_" + name + ".png"
        if os.path.exists(graph_loc):
            os.remove(graph_loc)

        val_name = "val_" + value
        plt.plot(history.history[value])
        plt.plot(history.history[val_name])
        plt.title('Complex NN {0} for {1} {2}'.format(name, set_name, future))
        plt.ylabel(name)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(graphs_directory + "/" + set_name + "_Complex_nn_" + str(future) + "_" + name + ".png")
    

def cnn_train_model(future, batch_size, epochs,
                model_directory, set_name, X_train, y_train, y_scaler):
    
    start_time = time.time()

    tuner = kt.Hyperband(cnn_kt_model, objective='mean_absolute_percentage_error', max_epochs=epochs, factor=3, 
                        directory=model_directory + "/" + set_name + "_kt_dir", project_name='kt_model_' + str(future), 
                        overwrite=True)

    monitor = EarlyStopping(monitor='mean_absolute_percentage_error', min_delta=1, patience=5, verbose=0, mode='auto', 
                    restore_best_weights=True)

    tuner.search(X_train, y_train, verbose=0, epochs=epochs, validation_split=0.2, batch_size=batch_size,
                callbacks=[monitor])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    # Split on a 3 monthly basis
    tss = TimeSeriesSplit(n_splits=10, test_size=48*90, gap=0)
    fold = 0
    total_metrics = {}

    for train_idx, val_idx in tss.split(X_train, y_train):

        fold_name = "Fold_" + str(fold)
        X_t = X_train[train_idx]
        X_v = X_train[val_idx]
        y_t = y_train[train_idx]
        y_v = y_train[val_idx]
        
        if fold == 9:
            history = model.fit(X_t, y_t, verbose=1, epochs=epochs, callbacks=[monitor],
                    batch_size=batch_size, validation_data=(X_v, y_v))
            graphs_directory = os.getcwd() + "/graphs"
            cnn_save_plots(history, graphs_directory, set_name, future)
            model.save(model_directory + "/" + set_name + "_Complex_nn_" + str(future))
        
        model.fit(X_t, y_t, verbose=0, epochs=epochs, callbacks=[monitor],
                    batch_size=batch_size)
        preds = model.predict(X_v, verbose=0)
        preds = y_scaler.inverse_transform(preds)
        metrics = cnn_get_metrics(preds, y_v, 1)
        total_metrics[fold_name] = metrics

        fold += 1

    end_time = time.time()
    total_time = start_time - end_time
    cnn_cross_val_metrics(total_metrics, set_name, future)

    return total_time


def cnn_predict(future, set_name, pred_dates_test, X_test, y_test, y_scaler, time):

    folder_path = os.getcwd()
    model_directory = folder_path + r"\models"
    csv_directory = folder_path + r"\csvs"

    model = load_model(model_directory + "/" + set_name + "_Complex_nn_" + str(future))
    predictions = model.predict(X_test)
    predictions = y_scaler.inverse_transform(predictions).reshape(-1)
    y_test = y_scaler.inverse_transform(y_test).reshape(-1)

    cnn_make_csvs(csv_directory, predictions, y_test, pred_dates_test, set_name, future, time)
    print(f"Finished running complex prediction on future window {0}", future)


def cnn_evaluate(future, set_name, X_train, y_train, epochs, batch_size, y_scaler):

    folder_path = os.getcwd()
    model_directory = folder_path + r"\models"

    absl.logging.set_verbosity(absl.logging.ERROR)
    tf.compat.v1.logging.set_verbosity(30)

    cnn_train_model(future, batch_size, epochs,
            model_directory, set_name, X_train, y_train, y_scaler)
    
    print("Finished evaluating complex nn for future {0}".format(future))