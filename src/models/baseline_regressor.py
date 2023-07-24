import os
import time

import absl.logging
import tensorflow as tf
from keras.layers import Dense
from keras.losses import MeanSquaredError
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error as mse

from models.configuration.baseline_regressor_config import BaselineRegressorConfig
from models.i_model import IModel
from performance_analysis import make_csvs, get_metrics


class BaselineRegressor(IModel):
    MODEL_NAME = "Baseline"

    @staticmethod
    def build_model():
        model = Sequential()
        model.add(Dense(16, activation="relu"))
        model.add(Dense(1, "linear"))

        model.compile(
            loss=MeanSquaredError(),
            optimizer=Adam(learning_rate=0.001),
            metrics=["mean_squared_error"],
        )

        return model

    @staticmethod
    def build_simple_model():
        model = Sequential()
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(16, activation="relu"))

        model.add(Dense(1, "linear"))

        model.compile(
            loss=MeanSquaredError(),
            optimizer=Adam(learning_rate=0.001),
            metrics=["mean_squared_error"],
        )

        return model

    def train(self, X_train, y_train, y_scaler, config: BaselineRegressorConfig):
        """
        _description_

        :param X_train: _description_
        :param y_train: _description_
        :param y_scaler: _description_
        :param config: _description_
        :return: _description_
        """
        # TODO: Ensure this is performed before passing to this train function
        # length = X_frame.shape[0]
        # X_train = X_frame[: int(length * split), :]
        # y_train = y_frame[: int(length * split)]

        # X_test = X_frame[int(length * split) :, :]
        # y_test = y_frame[int(length * split) :]

        model = self.build_model()
        model.fit(
            X_train,
            y_train,
            verbose=0,
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_split=0.2,
        )
        return model

    def predict(self, model, X_test, y_test, y_scaler, config: BaselineRegressorConfig):
        """
        _description_

        :param model: _description_
        :param X_test: _description_
        :param y_test: _description_
        :param y_scaler: _description_
        :param config: _description_
        :return: _description_
        """
        folder_path = os.getcwd()
        model_directory = folder_path + r"\models"
        csv_directory = folder_path + r"\csvs"

        model = load_model(
            f"{model_directory}/{config.set_name}_{self.MODEL_NAME}_{config.future}"
        )
        predictions = model.predict(X_test)
        predictions = y_scaler.inverse_transform(predictions).reshape(-1)
        y_test = y_scaler.inverse_transform(y_test).reshape(-1)

        # MA: Not sure if this is right, but could use something like this to get the dates
        pred_dates_test = X_test.index.dt.strftime("%Y-%m-%d").values

        make_csvs(
            csv_directory,
            predictions,
            y_test,
            pred_dates_test,
            config.set_name,
            config.future,
            self.MODEL_NAME,
        )

        print(
            "Finished running baseline prediction on future window {0}".format(future)
        )

        metric_outputs = get_metrics(predictions, y_test, 0, "Baseline")
        return metric_outputs

    def evaluate(self, X_train, y_train, y_scaler, config: BaselineRegressorConfig):
        """
        _description_

        :param X_train: _description_
        :param y_train: _description_
        :param y_scaler: _description_
        :param config: _description_
        :return: _description_
        """
        model_directory = os.getcwd() + "/models"

        time_start = time.time()

        folder_path = os.getcwd()
        model_directory = folder_path + r"\models"

        absl.logging.set_verbosity(absl.logging.ERROR)
        tf.compat.v1.logging.set_verbosity(30)

        model = self.train(X_train, y_train, y_scaler, config)
        model.save(
            f"{model_directory}/{config.set_name}_{self.MODEL_NAME}_{config.future}"
        )

        print(f"Finished evaluating baseline for future {config.future}")

        time_end = time.time()

        return time_end - time_start
