import math
from datacleaner import *
from basic_nn import *
from complex_nn import *
from xgboost_model import *
# from tpot_model import *
from rf_model import *
from performance_analysis import normalise_metrics, make_metrics_csvs


if __name__=="__main__":

    folder_path = os.getcwd()
    csv_directory = folder_path + r"\csvs"

    # These are the values that need changing per different dataset
    file_path = csv_directory + "/matlab_temp.xlsx"
    set_name = "matlab"
    target = "SYSLoad"
    trend_type = "Additive"
    epd = 48
    future = 1

    cleaning = 1
    training = 1
    predicting = 1
    eval_tpot = 0

    partition = 5000
    data_epochs = 10

    # Don't know what else to put in cleaning parameters tbh
    # put more research into this area
    cleaning_parameters = {
        # This is seriously jank. It works for now, but golly...
        'pca_dimensions': [None, math.inf, -math.inf],
        'scalers': ['standard', 'minmax']
    }

    window = 10
    split = 0.8
    epochs = 100
    batch_size = 32

    if cleaning:

        data, outputs = feature_adder(csv_directory, file_path, target, trend_type, future, epd,  set_name)

        # Decide on exactly what size this partition should be
        # Essentially this grid search is shit and has to be optimized later on down the track
        # It'll just get the job done for now
        best_results = data_cleaning_pipeline(data[:partition], outputs[:partition], cleaning_parameters, target, split, data_epochs, batch_size, csv_directory)

    else:

        if os.path.exists(csv_directory + "/best_data_parameters.csv") \
                and os.path.exists(csv_directory + "/" + set_name + "_data_" + str(future) + ".csv") \
                and os.path.exists(csv_directory + "/" + set_name + "_outputs_" + str(future) + ".csv"):
            
            best_results = pd.read_csv(csv_directory + "/best_data_parameters.csv").to_dict('index')
            best_results = best_results.get(0)
            data, outputs = load_datasets(csv_directory, set_name, future)

        else:
            data, outputs = feature_adder(csv_directory, file_path, target, trend_type, future, epd,  set_name)

            # Decide on exactly what size this partition should be
            # Essentially this grid search is shit and has to be optimized later on down the track
            # It'll just get the job done for now
            best_results = data_cleaning_pipeline(data[:partition], outputs[:partition], cleaning_parameters, target, split, data_epochs, batch_size, csv_directory)

    print("finished cleaning")
    X_frame, y_data, pred_dates, y_scaler = finalise_data(data, outputs, target, best_results)
    length = X_frame.shape[0]

    pred_dates_test = pred_dates[int(length*split) + window:]

    X_2d = create_dataset_2d(X_frame, window)
    X_3d = create_dataset_3d(X_frame, window)

    y_test = y_data[int(length*split) + window:]
    X_test_2d = X_2d[int(length*split):]
    X_test_3d = X_3d[int(length*split):]

    y_train = y_data[window:int(length*split) + window]
    X_train_2d = X_2d[:int(length * split)]
    X_train_3d = X_3d[:int(length * split)]

    if training:

        np.save("X_train_3d.npy", X_train_3d)

        bnn_time = bnn_evaluate(future, set_name, X_train_2d, y_train, epochs, batch_size, y_scaler, epd)
        cnn_time = cnn_evaluate(future, set_name, X_train_3d, y_train, epochs, batch_size, y_scaler, epd)
        xgb_time = xgb_evaluate(future, set_name, X_train_2d, y_train, epochs, epd)
        rf_time = rf_evaluate(future, set_name, X_train_2d, y_train.reshape(-1), epochs, epd)
        base_time = simple_evaluate(future, set_name, X_train_2d, y_train, epochs, batch_size)

    if predicting:

        bnn_metrics = bnn_predict(future, set_name, pred_dates_test, X_test_2d, y_test, y_scaler)
        cnn_metrics = cnn_predict(future, set_name, pred_dates_test, X_test_3d, y_test, y_scaler)
        xgb_metrics = xgb_predict(future, set_name, pred_dates_test, X_test_2d, y_test, y_scaler)
        rf_metrics = rf_predict(future, set_name, pred_dates_test, X_test_2d, y_test, y_scaler)
        base_metrics = simple_predict(future, set_name, pred_dates_test, X_test_2d, y_test, y_scaler)

        if training:
            bnn_metrics['TIME'] = bnn_time
            cnn_metrics['TIME'] = cnn_time
            xgb_metrics['TIME'] = xgb_time
            rf_metrics['TIME'] = rf_time
            base_metrics['TIME'] = base_time

        metrics = [bnn_metrics, cnn_metrics, xgb_metrics, rf_metrics, base_metrics]
        metrics = normalise_metrics(metrics, training)

        metrics = {"Basic_nn": metrics[0], "Complex_nn": metrics[1], "xgb": metrics[2], 
                    "rf": metrics[3], "Baseline": metrics[4]}

        make_metrics_csvs(csv_directory, metrics, set_name, future, training)

    if eval_tpot:
        # Problem still exists with tpot for whatever reason
        tpot_evaluate(future, set_name, X_train_2d, y_train.reshape(-1), pred_dates_test, X_test_2d, y_test.reshape(-1), y_scaler)

    if os.path.exists("X_train_3d.npy"):
        os.remove("X_train_3d.npy")
