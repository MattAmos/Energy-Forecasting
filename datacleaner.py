from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import pandas as pd
import numpy as np
import os

class FeatureScaler(BaseEstimator, TransformerMixin):

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


def scaling(csv_directory, future, set_name):

    data = pd.read_csv(csv_directory + "/" + set_name + "_data_" + str(future) + ".csv").drop('Date', axis=1)
    outputs = pd.read_csv(csv_directory + "/" + set_name + "_outputs_" + str(future) + ".csv")

    pipe = Pipeline([('Scaler', FeatureScaler())])
    scalers = pipe.fit_transform(data)

    pred_dates = outputs['Date']

    y_scaler = MinMaxScaler(feature_range=(0,1))

    y_data = y_scaler.fit_transform(outputs[['SYSLoad']])

    X_frame = np.array(data)
    y_data = np.array(y_data)

    return X_frame, y_data, pred_dates, y_scaler, scalers


def feature_adder(data, holidays, future, csv_directory, set_name):

    data['Holiday'] = data.index.isin(holidays['Date']).astype(int)
    data['PrevDaySameHour'] = data['SYSLoad'].copy().shift(48)
    data['PrevWeekSameHour'] = data['SYSLoad'].copy().shift(48*7)
    data['Prev24HourAveLoad'] = data['SYSLoad'].copy().rolling(window=48*7, min_periods=1).mean()
    data['Weekday'] = data.index.dayofweek
    data.loc[(data['Weekday'] < 5) & (data['Holiday'] == 0), 'IsWorkingDay'] = 1
    data.loc[(data['Weekday'] > 4) | (data['Holiday'] == 1), 'IsWorkingDay'] = 0
    data = data.dropna(how='any', axis='rows')

    y = data['SYSLoad'].shift(-48*future).reset_index(drop=True)
    y = y.dropna(how='any', axis='rows')

    future_dates = pd.Series(data.index[future*48:])
    outputs = pd.DataFrame({"Date": future_dates, "SYSLoad": y})

    if future > 10:
        data = data[['DryBulb', 'DewPnt', 'Prev5DayHighAve', 'Prev5DayLowAve', 'Hour', 'Weekday', 'IsWorkingDay']]
    else:
        data = data[['DryBulb', 'DewPnt', 'WetBulb','Humidity','Hour', 'Weekday', 'IsWorkingDay', 'PrevWeekSameHour', 'PrevDaySameHour', 'Prev24HourAveLoad']]

    data_name = csv_directory + "/" + set_name + "_data_" + str(future) + ".csv"
    output_name = csv_directory + "/" + set_name + "_outputs_" + str(future) + ".csv"

    data.to_csv(data_name)
    outputs.to_csv(output_name, index=False)

    print(f"Saved future window {0} to csvs", future)



if __name__=="__main__":

    try:
        
        folder_path = os.getcwd()
        csv_directory = folder_path + r"\csvs"
        
        data = pd.read_excel(csv_directory + r'\ausdata.xlsx').set_index("Date")
        holidays = pd.read_excel(csv_directory + r'\Holidays2.xls')

        feature_adder(data, holidays, 0, csv_directory, "matlab")
        feature_adder(data, holidays, 1, csv_directory, "matlab")
        feature_adder(data, holidays, 7, csv_directory, "matlab")
    
    except FileNotFoundError:

        print("Ausdata and Holidays2 xl files are not present in \"csvs\" directory.")
        print("Ensure they are before continuing")
