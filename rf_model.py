import absl.logging
from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error as mae, mean_absolute_percentage_error as mape
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Categorical
from skopt import dump, load
from statsmodels.tsa.seasonal import seasonal_decompose

import os
import pandas as pd
import numpy as np
    

def rf_get_metrics(predictions, actual, cv):

    MSE = mse(actual, predictions, squared=True)
    MAE = mae(actual, predictions)
    MAPE = mape(actual, predictions)
    RMSE = mse(actual, predictions, squared=False)
    R2 = r2_score(actual, predictions)
    if cv:
        metrics = {'RF_RMSE': RMSE, 'RF_R2': R2, 'RF_MSE': MSE, 'RF_MAE': MAE, 'RF_MAPE': MAPE}
    else:
        metrics = {'RMSE': RMSE, 'R2': R2, 'MSE': MSE, 'MAE': MAE, 'MAPE': MAPE}
    return metrics


def rf_cross_val_metrics(total_metrics, set_name, future):

    csv_directory = os.getcwd() + "/csvs"
    df = pd.DataFrame(total_metrics)
    df.to_csv(csv_directory + "/" + set_name + "_cv_metrics_" + str(future) + ".csv", index=False)


def rf_make_csvs(csv_directory, predictions, y_test, pred_dates_test, set_name, future):

    metric_outputs = rf_get_metrics(predictions, y_test, 0)

    if not os.path.exists(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv"):
        performances = pd.DataFrame({"Date":pred_dates_test, "Actual": y_test, "rf": predictions})
        performances.to_csv(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv", index=False)
    else:
        performances = pd.read_csv(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv")
        performances['rf'] = predictions
        performances.to_csv(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv", index=False)

    if not os.path.exists(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv"):
        new_row = {'Model': ["rf"], 'RMSE': [metric_outputs.get("RMSE")], 'R2': [metric_outputs.get("R2")], 
                    'MSE': [metric_outputs.get("MSE")], 'MAE': [metric_outputs.get("MAE")], 
                    'MAPE': [metric_outputs.get("MAPE")]}

        metrics = pd.DataFrame(new_row)
        metrics.to_csv(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv", index=False)
    else:

        metrics = pd.read_csv(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv")

        if 'rf' in metrics['Model'].values:
            metrics.loc[metrics['Model'] == 'rf', 'RMSE'] = metric_outputs.get("RMSE")
            metrics.loc[metrics['Model'] == 'rf', 'R2'] = metric_outputs.get("R2")
            metrics.loc[metrics['Model'] == 'rf', 'MSE'] = metric_outputs.get("MSE")
            metrics.loc[metrics['Model'] == 'rf', 'MAE'] = metric_outputs.get("MAE")
            metrics.loc[metrics['Model'] == 'rf', 'MAPE'] = metric_outputs.get("MAPE")
        else:
            new_row = {'Model': "rf", 'RMSE': metric_outputs.get("RMSE"), 'R2': metric_outputs.get("R2"), 
                        'MSE': metric_outputs.get("MSE"), 'MAE': metric_outputs.get("MAE"), 
                        'MAPE': metric_outputs.get("MAPE")}

            metrics.loc[len(metrics)] = new_row
        metrics.to_csv(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv", index=False)
    

def rf_train_model(future, epochs, model_directory, set_name, X_train, y_train, epd):

    tss = TimeSeriesSplit(n_splits=5, test_size=epd*90, gap=0)
    estimator = RandomForestRegressor()

    search_space = {
        "max_depth": (10, 1200),
        "min_samples_leaf": (0.001, 0.5, "uniform"),
        "min_samples_split": (0.001, 1.0, "uniform"),
        "n_estimators": (5, 5000),
        "criterion": Categorical(["squared_error"]),
        "max_features": Categorical(['sqrt', 'log2', None]),
    }

    model = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_space,
        scoring="neg_root_mean_squared_error",
        cv=tss,
        n_jobs=-1,
        n_iter=epochs,
        verbose=0,
        refit=True,
    )

    model = model.fit(X_train, y_train)
    dump(model, model_directory + "/" + set_name + "_rf_" + str(future) + ".pkl")


def rf_predict(future, set_name, pred_dates_test, X_test, y_test, y_scaler):

    folder_path = os.getcwd()
    model_directory = folder_path + r"\models"
    csv_directory = folder_path + r"\csvs"

    model = load(model_directory + "/" + set_name + "_rf_" + str(future) + ".pkl")
    predictions = model.predict(X_test).reshape(-1, 1)
    print(predictions.shape)
    predictions = y_scaler.inverse_transform(predictions)
    y_test = y_scaler.inverse_transform(y_test)

    rf_make_csvs(csv_directory, predictions, y_test, pred_dates_test, set_name, future)
    print(f"Finished running rf prediction on future window {0}", future)


def rf_evaluate(future, set_name, X_train, y_train, epochs, epd):

    folder_path = os.getcwd()
    model_directory = folder_path + r"\models"

    absl.logging.set_verbosity(absl.logging.ERROR)

    rf_train_model(future, epochs,
            model_directory, set_name, X_train, y_train, epd)
    
    print("Finished evaluating basic for future {0}".format(future))
    
