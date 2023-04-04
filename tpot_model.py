import absl.logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error as mae, mean_absolute_percentage_error as mape
from statsmodels.tsa.seasonal import seasonal_decompose

from tpot import TPOTRegressor

import os
import pandas as pd
import numpy as np


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

    X = np.array(X)
    X = X.flatten().reshape(X.shape[0], -1)

    return X



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
        performances = pd.DataFrame({"Date":pred_dates_test, "Actual": y_test, "TPOT": predictions})
        performances.to_csv(csv_directory + r"\performances_" + str(future) + ".csv", index=False)
    else:
        performances = pd.read_csv(csv_directory + r"\performances_" + str(future) + ".csv")
        performances['TPOT'] = predictions
        performances.to_csv(csv_directory + r"\performances_" + str(future) + ".csv", index=False)

    if not os.path.exists(csv_directory + r"\metrics_" + str(future) + ".csv"):
        new_row = {'Model': ["TPOT"], 'RMSE': [metric_outputs.get("RMSE")], 'R2': [metric_outputs.get("R2")], 
                    'MSE': [metric_outputs.get("MSE")], 'MAE': [metric_outputs.get("MAE")], 
                    'MAPE': [metric_outputs.get("MAPE")]}

        metrics = pd.DataFrame(new_row)
        metrics.to_csv(csv_directory + r"\metrics_" + str(future) + ".csv", index=False)
    else:

        metrics = pd.read_csv(csv_directory + r"\metrics_" + str(future) + ".csv")

        if 'TPOT' in metrics['Model'].values:
            metrics.loc[metrics['Model'] == 'TPOT', 'RMSE'] = metric_outputs.get("RMSE")
            metrics.loc[metrics['Model'] == 'TPOT', 'R2'] = metric_outputs.get("R2")
            metrics.loc[metrics['Model'] == 'TPOT', 'MSE'] = metric_outputs.get("MSE")
            metrics.loc[metrics['Model'] == 'TPOT', 'MAE'] = metric_outputs.get("MAE")
            metrics.loc[metrics['Model'] == 'TPOT', 'MAPE'] = metric_outputs.get("MAPE")
        else:
            new_row = {'Model': "TPOT", 'RMSE': metric_outputs.get("RMSE"), 'R2': metric_outputs.get("R2"), 
                        'MSE': metric_outputs.get("MSE"), 'MAE': metric_outputs.get("MAE"), 
                        'MAPE': metric_outputs.get("MAPE")}

            metrics = metrics.append(new_row, ignore_index=True)
        metrics.to_csv(csv_directory + r"\metrics_" + str(future) + ".csv", index=False)
    

def train_model(X_frame, y_data, window, future, batch_size, split, epochs, pred_dates, y_scaler,
                model_directory, csv_directory):

    X_frame, y_data, pred_dates, y_scaler, scalers = scaling(csv_directory, future)
    length = X_frame.shape[0]

    y_train = y_data[window:int(length*split)]

    X_train = create_dataset(X_frame[:int(length*split)], window)

    X_train.shape, y_train.shape

    model = TPOTRegressor(early_stop=5, verbosity=1, max_time_mins=180)
    model.fit(X_train, y_train)
    model.export(csv_directory + r"\tpot_" + str(future) + ".py")

    print("Completed generating tpot model")

    # predictions = model.predict(X_test)
    # predictions = y_scaler.inverse_transform(predictions).reshape(-1)
    # y_test = y_scaler.inverse_transform(y_test).reshape(-1)

    # make_csvs(csv_directory, predictions, y_test, pred_dates_test)
    # print(f"Finished running simulation on future window {0}", future)


if __name__=='__main__':

    folder_path = os.getcwd()
    model_directory = folder_path + r"\models"
    csv_directory = folder_path + r"\csvs"

    standard_scaling = 1
    epochs = 2
    batch_size = 1
    future = 0
    window = 3
    split = 0.7

    absl.logging.set_verbosity(absl.logging.ERROR)

    try:

        X_frame, y_data, pred_dates, y_scaler, scalers = scaling(csv_directory, future)

        train_model(X_frame, y_data, window, future, split, pred_dates, y_scaler, 
                model_directory, csv_directory)

    except FileNotFoundError:

        print("Files are not present in \"csvs\" directory.")
        print("Ensure they are before continuing")
    
