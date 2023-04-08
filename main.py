from datacleaner import *
from basic_nn import *
from complex_nn import *
from xgboost_model import *
from tpot_model import *
from rf_model import *


if __name__=="__main__":

    window = 10
    set_name = "matlab"
    split = 0.7
    epochs = 100
    batch_size = 32

    futures = [0, 1, 7]

    folder_path = os.getcwd()
    csv_directory = folder_path + r"\csvs"

    for future in futures:

        if not os.path.exists(csv_directory + "/" + set_name + "_data_" + str(future) + ".csv"):
            feature_adder(data, holidays, future, csv_directory, set_name)

        X_frame, y_data, pred_dates, y_scaler, _ = scaling(csv_directory, future, set_name)
        length = X_frame.shape[0]

        pred_dates_test = pred_dates[int(length*split) + window:]

        y_test = y_data[int(length*split) + window:]
        X_test_2d = create_dataset_2d(X_frame[int(length*split):], window)
        X_test_3d = create_dataset_3d(X_frame[int(length*split):], window)

        y_train = y_data[window:int(length*split)]
        X_train_2d = create_dataset_2d(X_frame[:int(length*split)], window)
        X_train_3d = create_dataset_3d(X_frame[:int(length*split)], window)

        np.save("X_train_3d.npy", X_train_3d)

        bnn_time = bnn_evaluate(future, set_name, X_train_2d, y_train, epochs, batch_size, y_scaler)
        bnn_predict(future, set_name, pred_dates_test, X_test_2d, y_test, y_scaler, bnn_time)

        cnn_time = cnn_evaluate(future, set_name, X_train_3d, y_train, epochs, batch_size, y_scaler)
        cnn_predict(future, set_name, pred_dates_test, X_test_3d, y_test, y_scaler, cnn_time)

        xgb_time = xgb_evaluate(future, set_name, X_train_2d, y_train, epochs, y_scaler)
        xgb_predict(future, set_name, pred_dates_test, X_test_2d, y_test, y_scaler, xgb_time)

        rf_time = rf_evaluate(future, set_name, X_train_2d, y_train.reshape(-1), epochs, y_scaler)
        rf_predict(future, set_name, pred_dates_test, X_test_2d, y_test, y_scaler, rf_time)

        # Making tpot predictions has not yet been implemented
        tpot_time = tpot_evaluate(future, set_name, X_train_2d, y_train.reshape(-1))

        os.remove("X_train_3d.npy")
