from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from statsmodels.tsa.seasonal import seasonal_decompose

import pandas as pd
import numpy as np
import os
import math

from simple_regression import *

def collapse_columns(data):
    data = data.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.to_series().apply(lambda x: "__".join(x))
    return data
    
def create_dataset_2d(input, win_size):
    
    np_data = input.copy()

    X = []

    for i in range(len(np_data)-win_size):
        row = [r for r in np_data[i:i+win_size]]
        X.append(row)

    X = np.array(X)
    X = X.reshape(X.shape[0], -1)

    return X
    

def create_dataset_3d(input, win_size):
    
    np_data = input.copy()

    X = []

    for i in range(len(np_data)-win_size):
        row = [r for r in np_data[i:i+win_size]]
        X.append(row)

    return np.array(X)


def finalise_data(data, outputs, target, best_results):

    pca_dim = best_results.get("pca_dimensions")
    y_scaler = None
    
    if best_results.get("scaler") == "minmax":
        X_scaler = MinMaxScaler(feature_range=(0,1))
        y_scaler = MinMaxScaler(feature_range=(0,1))
        data = X_scaler.fit_transform(data)
        outputs = y_scaler.fit_transform(outputs)

    elif best_results.get("scaler") == "standard":
        X_scaler = StandardScaler(feature_range=(0,1))
        y_scaler = StandardScaler(feature_range=(0,1))
        data = X_scaler.fit_transform(data)
        outputs = y_scaler.fit_transform(outputs)

    if pca_dim == 0:
        pca = PCA()
        data = pca.fit_transform(data)
    elif pca_dim != math.inf:
        pca = PCA(n_components=pca_dim)
        data = pca.fit_transform(data)

    X_frame = np.array(data)
    y_data = np.array(outputs[target])
    pred_dates = outputs['Date']

    return X_frame, y_data, pred_dates, y_scaler


def data_cleaning_pipeline(data_in, outputs_in, cleaning_parameters, target):

    best_results = {"MSE": [math.inf], "scaler": [None], "pca_dimensions": [None]}
    model = build_simple_model()

    for scale_type in cleaning_parameters.get('scalers'):
            for pca_dim in cleaning_parameters.get('pca_dimensions'):

                data = data_in.copy()
                outputs = outputs_in[target].copy()

                if scale_type == 'minmax':
                    X_scaler = MinMaxScaler(feature_range=(0,1))
                    y_scaler = MinMaxScaler(feature_range=(0,1))
                    data = X_scaler.fit_transform(data)
                    outputs = y_scaler.fit_transform(outputs)

                elif scale_type == 'standard':
                    X_scaler = StandardScaler(feature_range=(0,1))
                    y_scaler = StandardScaler(feature_range=(0,1))
                    data = X_scaler.fit_transform(data)
                    outputs = y_scaler.fit_transform(outputs)

                if pca_dim == 0:
                    pca = PCA()
                    data = pca.fit_transform(data)
                elif pca_dim != math.inf:
                    pca = PCA(n_components=pca_dim)
                    data = pca.fit_transform(data)
                
                mse = train_simple_model(model, np.array(data), np.array(outputs))
                if mse < best_results.get("MSE"):
                    best_results["MSE"][0] = mse
                    best_results["pca_dimensions"][0] = pca_dim
                    best_results["scaler"][0] = scale_type

    results_data = pd.DataFrame.from_dict(best_results)
    results_data.to_csv(csv_directory + "/best_data_parameters.csv", index=False)

    return best_results




def feature_adder(csv_directory, file_path, target, trend_type, future, epd,  set_name):

    data = pd.read_excel(file_path).set_index("Date")
    data = collapse_columns(data)

    data['PrevDaySameHour'] = data[target].copy().shift(epd)
    data['PrevWeekSameHour'] = data[target].copy().shift(epd*7)
    data['Prev24HourAveLoad'] = data[target].copy().rolling(window=epd*7, min_periods=1).mean()
    data['Weekday'] = data.index.dayofweek
    data.loc[(data['Weekday'] < 5) & (data['Holiday'] == 0), 'IsWorkingDay'] = 1
    data.loc[(data['Weekday'] > 4) | (data['Holiday'] == 1), 'IsWorkingDay'] = 0

    dec_daily = seasonal_decompose(data[target], model=trend_type, period=epd)
    dec_weekly = seasonal_decompose(data[target], model=trend_type, period=epd*7)

    data['IntraDayTrend'] = dec_daily.trend
    data['IntraDaySeasonal'] = dec_daily.seasonal
    data['IntraWeekTrend'] = dec_weekly.trend
    data['IntraWeekSeasonal'] = dec_weekly.seasonal

    data = data.dropna(how='any', axis='rows')
    y = data[target].shift(-epd*future).reset_index(drop=True)
    y = y.dropna(how='any', axis='rows')

    future_dates = pd.Series(data.index[future*epd:])
    outputs = pd.DataFrame({"Date": future_dates, "{0}".format(target): y})

    # future > 10 needs addressing - it is not yet implemented
    if future > 10:
        data = data[['DryBulb', 'DewPnt', 'Prev5DayHighAve', 'Prev5DayLowAve', 'Hour', 'Weekday', 'IsWorkingDay']]
    else:
        data = data.drop("{0}".format(target), axis=1)

    data_name = csv_directory + "/" + set_name + "_data_" + str(future) + ".csv"
    output_name = csv_directory + "/" + set_name + "_outputs_" + str(future) + ".csv"

    data.to_csv(data_name)
    outputs.to_csv(output_name, index=False)

    print(f"Saved future window {0} to csvs", future)

    return data, outputs
    


if __name__=="__main__":

    try:
        
        folder_path = os.getcwd()
        csv_directory = folder_path + r"\csvs"
        
        data = pd.read_excel(csv_directory + r'\ausdata.xlsx').set_index("Date")
        holidays = pd.read_excel(csv_directory + r'\Holidays2.xls')

        data['Holiday'] = data.index.isin(holidays['Date']).astype(int)

        file_name = csv_directory + "/matlab_temp.xlsx"
        data.to_excel(file_name)
    
    except FileNotFoundError:

        print("Ausdata and Holidays2 xl files are not present in \"csvs\" directory.")
        print("Ensure they are before continuing")
