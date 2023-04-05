import pandas as pd
import os
import re

class availablemodes:

    def __init__(self):
        self.datasets = {}
        self.find_datasets()


    def get_keys(self):
        return list(self.datasets.keys())


    def get_models(self, dataset, directory):

        file_path = directory + "\\" + dataset
        frame = pd.read_csv(file_path)
        cols = frame.columns.values
        cols = [col for col in cols if col != "Date" and col != "Actual"]
        return cols
    

    def get_periods(self, dataset, directory):

        periods = []
        files = os.listdir(directory)
        files = " ".join(files)
        datasets = re.findall(" " + dataset + "_performances_[0-9]+.csv", files)

        for file in datasets:
            if dataset in file:
                periods.append(int(re.findall(r'\d+', file)[0]))

        return periods
                

    def find_datasets(self):

        folder_path = os.getcwd()
        csv_directory = folder_path + r"\csvs"
        files = os.listdir(csv_directory)
        files = " ".join(files)
        datasets = re.findall(" [a-zA-Z]+_performances_[0-9]+.csv", files)
        for dataset in datasets:
                set_name = str.split(dataset, '_')[0][1:]
                periods = self.get_periods(set_name, csv_directory)
                models = self.get_models(dataset[1:], csv_directory)
                self.datasets[set_name] = {"Periods": periods, "Models": models}

    def return_dataset(self, dataset):
        return self.datasets.get(dataset)


    


class infokeeper:

    def __init__(self):
        self.model = None
        self.baseline = 0
        self.period = 0

    def set_model(self, model):
        self.model = model

    def set_baseline(self, baseline):
        self.baseline = baseline

    def set_period(self, period):
        self.period = period

    def get_stats(self):
        return self.model, self.baseline, self.period


class datasets:

    def __init__(self):

        self.performance_0 = None
        self.performance_1 = None
        self.performance_7 = None

        self.metrics_0 = None
        self.metrics_0 = None
        self.metrics_0 = None

        self.get_datasets()


    def get_datasets(self):

        folder_path = os.getcwd()
        csv_directory = folder_path + r"\csvs"

        try:
            self.performance_0 = pd.read_csv(csv_directory + r"\performances_0.csv")

            self.metrics_0 = pd.read_csv(csv_directory + r"\metrics_0.csv")

        except FileNotFoundError:

            print("Didn't find all expected files")


    def get_performance_data(self, model, baseline, period):
        
        if period == 0:
            dataframe = self.performance_0
        elif period == 1:
            dataframe = self.performance_1
        elif period == 7:
            dataframe = self.performance_7

        if baseline:
            dataframe = dataframe[["Date", "Actual", model, "Baseline"]].iloc[-200:]
        else:
            dataframe = dataframe[["Date", "Actual", model]].iloc[-200:]

        return dataframe
    
    
    def get_metrics_data(self, model, baseline, period):
        
        if period == 0:
            dataframe = self.metrics_0
        elif period == 1:
            dataframe = self.metrics_1
        elif period == 7:
            dataframe = self.metrics_7

        if baseline:
            baseline_row = dataframe[dataframe['Model'] == "Baseline"]
            model_row = dataframe[dataframe['Model'] == model]
            rows = pd.concat([baseline_row, model_row], ignore_index=True)
        else:
            rows = dataframe[dataframe['Model'] == model]

        return rows