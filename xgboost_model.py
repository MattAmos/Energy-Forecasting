import absl.logging
from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error as mae, mean_absolute_percentage_error as mape
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt import dump, load
from statsmodels.tsa.seasonal import seasonal_decompose

import os
import pandas as pd
import numpy as np
import xgboost as xgb
    

def xgb_get_metrics(predictions, actual, cv):

    MSE = mse(actual, predictions, squared=True)
    MAE = mae(actual, predictions)
    MAPE = mape(actual, predictions)
    RMSE = mse(actual, predictions, squared=False)
    R2 = r2_score(actual, predictions)
    if cv:
        metrics = {'XGB_RMSE': RMSE, 'XGB_R2': R2, 'XGB_MSE': MSE, 'XGB_MAE': MAE, 'XGB_MAPE': MAPE}
    else:
        metrics = {'RMSE': RMSE, 'R2': R2, 'MSE': MSE, 'MAE': MAE, 'MAPE': MAPE}
    return metrics


def xgb_cross_val_metrics(total_metrics, set_name, future):

    csv_directory = os.getcwd() + "/csvs"
    df = pd.DataFrame(total_metrics)
    df.to_csv(csv_directory + "/" + set_name + "_cv_metrics_" + str(future) + ".csv", index=False)


def xgb_make_csvs(csv_directory, predictions, y_test, pred_dates_test, set_name, future):

    metric_outputs = xgb_get_metrics(predictions, y_test, 0)

    if not os.path.exists(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv"):
        performances = pd.DataFrame({"Date":pred_dates_test, "Actual": y_test, "xgb": predictions})
        performances.to_csv(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv", index=False)
    else:
        performances = pd.read_csv(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv")
        performances['xgb'] = predictions
        performances.to_csv(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv", index=False)

    if not os.path.exists(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv"):
        new_row = {'Model': ["xgb"], 'RMSE': [metric_outputs.get("RMSE")], 'R2': [metric_outputs.get("R2")], 
                    'MSE': [metric_outputs.get("MSE")], 'MAE': [metric_outputs.get("MAE")], 
                    'MAPE': [metric_outputs.get("MAPE")]}

        metrics = pd.DataFrame(new_row)
        metrics.to_csv(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv", index=False)
    else:

        metrics = pd.read_csv(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv")

        if 'xgb' in metrics['Model'].values:
            metrics.loc[metrics['Model'] == 'xgb', 'RMSE'] = metric_outputs.get("RMSE")
            metrics.loc[metrics['Model'] == 'xgb', 'R2'] = metric_outputs.get("R2")
            metrics.loc[metrics['Model'] == 'xgb', 'MSE'] = metric_outputs.get("MSE")
            metrics.loc[metrics['Model'] == 'xgb', 'MAE'] = metric_outputs.get("MAE")
            metrics.loc[metrics['Model'] == 'xgb', 'MAPE'] = metric_outputs.get("MAPE")
        else:
            new_row = {'Model': "xgb", 'RMSE': metric_outputs.get("RMSE"), 'R2': metric_outputs.get("R2"), 
                        'MSE': metric_outputs.get("MSE"), 'MAE': metric_outputs.get("MAE"), 
                        'MAPE': metric_outputs.get("MAPE")}

            metrics.loc[len(metrics)] = new_row
        metrics.to_csv(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv", index=False)
    

def xgb_train_model(future, epochs, model_directory, set_name, X_train, y_train, epd):
    
    split = 0.9

    length = X_train.shape[0]
    X_train_temp = X_train[:int(length * split), :]
    y_train_temp = y_train[:int(length * split), :]
    X_val = X_train[int(length * split):, :]
    y_val = y_train[int(length * split):, :]

    tss = TimeSeriesSplit(n_splits=5, test_size=epd*90, gap=0)
    estimator = xgb.XGBRegressor(booster='gbtree',    
            early_stopping_rounds=50,
            objective='reg:squarederror',
            verbosity=0)

    search_space = {
        "learning_rate": (0.01, 1.0, "log-uniform"),
        "min_child_weight": (0, 10),
        "max_depth": (1, 50),
        "subsample": (0.01, 1.0, "uniform"),
        "colsample_bytree": (0.01, 1.0, "log-uniform"),
        "reg_lambda": (1e-9, 1.0, "log-uniform"),
        "reg_alpha": (1e-9, 1.0, "log-uniform"),
        "gamma": (1e-9, 0.5, "log-uniform"),
        "n_estimators": (5, 5000),
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

    model = model.fit(X_train_temp, y_train_temp, eval_set=[(X_val, y_val)], verbose=False)
    dump(model, model_directory + "/" + set_name + "_xgb_" + str(future) + ".pkl")


def xgb_predict(future, set_name, pred_dates_test, X_test, y_test, y_scaler):

    folder_path = os.getcwd()
    model_directory = folder_path + r"\models"
    csv_directory = folder_path + r"\csvs"

    model = load(model_directory + "/" + set_name + "_xgb_" + str(future) + ".pkl")
    predictions = model.predict(X_test).reshape(-1, 1)
    print(predictions.shape)
    predictions = y_scaler.inverse_transform(predictions)
    y_test = y_scaler.inverse_transform(y_test)

    xgb_make_csvs(csv_directory, predictions, y_test, pred_dates_test, set_name, future)
    print(f"Finished running xgb prediction on future window {0}", future)


def xgb_evaluate(future, set_name, X_train, y_train, epochs, epd):

    folder_path = os.getcwd()
    model_directory = folder_path + r"\models"

    absl.logging.set_verbosity(absl.logging.ERROR)

    xgb_train_model(future, epochs,
            model_directory, set_name, X_train, y_train, epd)
    
    print("Finished evaluating basic for future {0}".format(future))
    
