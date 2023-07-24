from models.configuration.i_config import IConfig
from abc import ABC, abstractmethod


class IModel(ABC):
    """
    Interface for models to inherit from.

    :raises NotImplementedError: This class is not a concrete class and as such cannot be instantiated
    """

    MODEL_NAME = None

    @abstractmethod
    def train(self, X_train, y_train, y_scaler, config: IConfig):
        """
        Function to perform training against an input set of data to produce a set of forecasts.

        :param X_train: Training input data
        :param y_train: Training output (target) data
        :param y_scaler: Scaler to be used against the output (target) data
        :param config: Configuration for the model
        """
        pass

    @abstractmethod
    def predict(self, model, X_test, y_test, y_scaler, config: IConfig):
        """
        Function to perform predictions against an input set of data to produce a set of forecasts.

        :param model: Model to be used for prediction
        :param X_test: Test input data
        :param y_test: Test output (target) data
        :param y_scaler: Scaler to be used against the output (target) data
        :param config: Configuration for the model
        """
        pass

    @abstractmethod
    def evaluate(self, X_train, y_train, y_scaler, config: IConfig):
        """
        Function to perform evaluation against an input set of data to produce a set of forecasts.

        :param X_train: Training input data
        :param y_train: Training output (target) data
        :param y_scaler: Scaler to be used against the output (target) data
        :param config: Configuration for the model
        """
        pass
