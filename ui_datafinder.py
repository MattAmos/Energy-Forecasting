import pandas as pd
import os


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
            rows = {"Baseline": baseline_row, "Model": model_row}
        else:
            rows = dataframe[dataframe['Model'] == model]

        return rows