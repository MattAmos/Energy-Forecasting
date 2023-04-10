import absl.logging
from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error as mae, mean_absolute_percentage_error as mape
from tpot import TPOTRegressor
from skopt import dump, load
from statsmodels.tsa.seasonal import seasonal_decompose

import os
import pandas as pd
    
# 90% of this file is not needed, since it should be within the generated model export
# Going to have to figure out exactly how that part is done, tbh

def tpot_get_metrics(predictions, actual, cv):

    MSE = mse(actual, predictions, squared=True)
    MAE = mae(actual, predictions)
    MAPE = mape(actual, predictions)
    RMSE = mse(actual, predictions, squared=False)
    R2 = r2_score(actual, predictions)
    if cv:
        metrics = {'TPOT_RMSE': RMSE, 'TPOT_R2': R2, 'TPOT_MSE': MSE, 'TPOT_MAE': MAE, 'TPOT_MAPE': MAPE}
    else:
        metrics = {'RMSE': RMSE, 'R2': R2, 'MSE': MSE, 'MAE': MAE, 'MAPE': MAPE}
    return metrics


def tpot_cross_val_metrics(total_metrics, set_name, future):

    csv_directory = os.getcwd() + "/csvs"
    df = pd.DataFrame(total_metrics)
    df.to_csv(csv_directory + "/" + set_name + "_cv_metrics_" + str(future) + ".csv", index=False)


def tpot_make_csvs(csv_directory, predictions, y_test, pred_dates_test, set_name, future):

    metric_outputs = tpot_get_metrics(predictions, y_test, 0)

    if not os.path.exists(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv"):
        performances = pd.DataFrame({"Date":pred_dates_test, "Actual": y_test, "tpot": predictions})
        performances = performances.iloc[-1000:,:]
        performances.to_csv(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv", index=False)
    else:
        performances = pd.read_csv(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv")
        performances['tpot'] = predictions[-1000:]
        performances.to_csv(csv_directory + "/" + set_name + "_performances_" + str(future) + ".csv", index=False)

    if not os.path.exists(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv"):
        new_row = {'Model': ["tpot"], 'RMSE': [metric_outputs.get("RMSE")], 'R2': [metric_outputs.get("R2")], 
                    'MSE': [metric_outputs.get("MSE")], 'MAE': [metric_outputs.get("MAE")], 
                    'MAPE': [metric_outputs.get("MAPE")]}

        metrics = pd.DataFrame(new_row)
        metrics.to_csv(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv", index=False)
    else:

        metrics = pd.read_csv(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv")

        if 'tpot' in metrics['Model'].values:
            metrics.loc[metrics['Model'] == 'tpot', 'RMSE'] = metric_outputs.get("RMSE")
            metrics.loc[metrics['Model'] == 'tpot', 'R2'] = metric_outputs.get("R2")
            metrics.loc[metrics['Model'] == 'tpot', 'MSE'] = metric_outputs.get("MSE")
            metrics.loc[metrics['Model'] == 'tpot', 'MAE'] = metric_outputs.get("MAE")
            metrics.loc[metrics['Model'] == 'tpot', 'MAPE'] = metric_outputs.get("MAPE")
        else:
            new_row = {'Model': "tpot", 'RMSE': metric_outputs.get("RMSE"), 'R2': metric_outputs.get("R2"), 
                        'MSE': metric_outputs.get("MSE"), 'MAE': metric_outputs.get("MAE"), 
                        'MAPE': metric_outputs.get("MAPE")}

            metrics.loc[len(metrics)] = new_row
        metrics.to_csv(csv_directory + "/" + set_name + "_metrics_" + str(future) + ".csv", index=False)
    

def tpot_train_model(future, model_directory, set_name, X_train, y_train, pred_dates_test, X_test, y_test, y_scaler):

    model = TPOTRegressor(early_stop=5, verbosity=0, max_time_mins=1, cv=10)
    model.fit(X_train, y_train)

    tpot_predict(future, set_name, pred_dates_test, X_test, y_test, y_scaler)

    model.export(model_directory + "/" + set_name + "_tpot_" + str(future) + ".py")

    print("Completed generating tpot model")


def tpot_predict(future, set_name, pred_dates_test, X_test, y_test, y_scaler):

    folder_path = os.getcwd()
    model_directory = folder_path + r"\models"
    csv_directory = folder_path + r"\csvs"

    model = load(model_directory + "/" + set_name + "_tpot_" + str(future) + ".pkl")
    predictions = model.predict(X_test).reshape(-1, 1)
    print(predictions.shape)
    predictions = y_scaler.inverse_transform(predictions)
    y_test = y_scaler.inverse_transform(y_test)

    tpot_make_csvs(csv_directory, predictions, y_test, pred_dates_test, set_name, future)
    print(f"Finished running tpot prediction on future window {0}", future)


def tpot_evaluate(future, set_name, X_train, y_train, pred_dates_test, X_test, y_test, y_scaler):

    folder_path = os.getcwd()
    model_directory = folder_path + r"\models"

    absl.logging.set_verbosity(absl.logging.ERROR)

    tpot_train_model(future, model_directory, set_name, X_train, y_train, pred_dates_test, X_test, y_test, y_scaler)
    


    



