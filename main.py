from datacleaner import *
from basic_nn import *


if __name__=="__main__":

    window = 10
    set_name = "matlab"
    split = 0.7
    epochs = 1
    batch_size = 32

    futures = [0, 1, 7]

    folder_path = os.getcwd()
    csv_directory = folder_path + r"\csvs"

    for future in futures:

        X_frame, y_data, pred_dates, y_scaler, _ = scaling(csv_directory, future, set_name)
        length = X_frame.shape[0]

        pred_dates_test = pred_dates[int(length*split) + window:]

        y_train = y_data[window:int(length*split)]
        X_train_2d = create_dataset_2d(X_frame[:int(length*split)], window)
        X_train_3d = create_dataset_3d(X_frame[:int(length*split)], window)

        y_test = y_data[int(length*split) + window:]
        X_test_2d = create_dataset_2d(X_frame[int(length*split):], window)
        X_test_3d = create_dataset_3d(X_frame[int(length*split):], window)

        np.save("X_train_3d.npy", X_train_3d)

        time = bnn_evaluate(future, set_name, X_train_2d, y_train, epochs, batch_size)
        bnn_predict(future, set_name, pred_dates_test, X_test_2d, y_test, y_scaler, time)

        # cnn_evaluate(window, future, set_name)
        # cnn_predict(window, future, set_name, pred_dates_test, X_test, y_test, y_scaler)

        os.remove("X_train_3d.npy")
