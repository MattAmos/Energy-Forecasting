import absl.logging
from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error as mae, mean_absolute_percentage_error as mape
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from statsmodels.tsa.seasonal import seasonal_decompose

import os
import time
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


def xgb_make_csvs(csv_directory, predictions, y_test, pred_dates_test, set_name, future, time):

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
                    'MAPE': [metric_outputs.get("MAPE")], "TIME": [time]}

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
            metrics.loc[metrics['Model'] == 'xgb', 'TIME'] = time
        else:
            new_row = {'Model': "xgb", 'RMSE': metric_outputs.get("RMSE"), 'R2': metric_outputs.get("R2"), 
                        'MSE': metric_outputs.get("MSE"), 'MAE': metric_outputs.get("MAE"), 
                        'MAPE': metric_outputs.get("MAPE")}

            metrics = metrics.append(new_row, ignore_index=True)
        metrics.to_csv(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv", index=False)
    

def xgb_train_model(future, epochs, model_directory, set_name, X_train, y_train, y_scaler):
    
    start_time = time.time()

    tss = TimeSeriesSplit(n_splits=10, test_size=48*90, gap=0)
    estimator = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
            early_stopping_rounds=50,
            objective='reg:squarederror',
            verbosity=0)

    search_space = {
        "learning_rate": (0.01, 1.0, "log-uniform"),
        "min_child_weight": (0, 10),
        "max_depth": (1, 50),
        "max_delta_step": (0, 10),
        "subsample": (0.01, 1.0, "uniform"),
        "colsample_bytree": (0.01, 1.0, "log-uniform"),
        "colsample_bylevel": (0.01, 1.0, "log-uniform"),
        "reg_lambda": (1e-9, 1000, "log-uniform"),
        "reg_alpha": (1e-9, 1.0, "log-uniform"),
        "gamma": (1e-9, 0.5, "log-uniform"),
        "min_child_weight": (0, 5),
        "n_estimators": (5, 5000),
        "scale_pos_weight": (1e-6, 500, "log-uniform"),
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

    model.fit(X_train, y_train)
    model.save_model(model_directory + "/" + set_name + "_xgb_" + str(future) + ".txt")

    model = xgb.XGBRegressor()
    model.load_model(model_directory + "/" + set_name + "_xgb_" + str(future) + ".txt")

    fold = 0
    total_metrics = {}

    for train_idx, val_idx in tss.split(X_train, y_train):

        fold_name = "Fold_" + str(fold)
        X_t = X_train[train_idx]
        X_v = X_train[val_idx]
        y_t = y_train[train_idx]
        y_v = y_train[val_idx]
        
        model.fit(X_t, y_t,  verbosity=0)
        preds = model.predict(X_v)
        preds = y_scaler.inverse_transform(preds)
        metrics = xgb_get_metrics(preds, y_v, 1)
        total_metrics[fold_name] = metrics

        fold += 1

    end_time = time.time()
    total_time = start_time - end_time
    xgb_cross_val_metrics(total_metrics, set_name, future)
    model.save_model(model_directory + "/" + set_name + "_xgb_" + str(future) + ".txt")

    return total_time


def xgb_predict(future, set_name, pred_dates_test, X_test, y_test, y_scaler, time):

    folder_path = os.getcwd()
    model_directory = folder_path + r"\models"
    csv_directory = folder_path + r"\csvs"

    model = xgb.XGBRegressor()
    model.load_model(model_directory + "/" + set_name + "_xgb_" + str(future) + ".txt")
    predictions = model.predict(X_test)
    print(predictions.shape)
    predictions = y_scaler.inverse_transform(predictions).reshape(-1)
    y_test = y_scaler.inverse_transform(y_test).reshape(-1)

    xgb_make_csvs(csv_directory, predictions, y_test, pred_dates_test, set_name, future, time)
    print(f"Finished running basic prediction on future window {0}", future)


def xgb_evaluate(future, set_name, X_train, y_train, epochs, y_scaler):

    folder_path = os.getcwd()
    model_directory = folder_path + r"\models"

    absl.logging.set_verbosity(absl.logging.ERROR)

    xgb_train_model(future, epochs,
            model_directory, set_name, X_train, y_train, y_scaler)
    
    print("Finished evaluating basic for future {0}".format(future))
    
