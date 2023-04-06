import pandas as pd
import numpy as np
import os

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
