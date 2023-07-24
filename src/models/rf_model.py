import os
import time

import absl.logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV, dump, load
from skopt.space import Categorical

from models.configuration.rf_config import RFConfig
from models.i_model import IModel
from performance_analysis import get_metrics, make_csvs


class RFModel(IModel):
    """
    _description_
    """

    MODEL_NAME = "RF"

    def train(self, X_train, y_train, y_scaler, config: RFConfig):
        """
        _description_

        :param X_train: _description_
        :param y_train: _description_
        :param y_scaler: _description_
        :param config: _description_
        """
        tss = TimeSeriesSplit(n_splits=5, test_size=config.epd * 90, gap=0)
        estimator = RandomForestRegressor()

        search_space = {
            "max_depth": (config.max_depth_low, config.max_depth_high),
            "min_samples_leaf": (
                config.min_samples_leaf_low,
                config.min_samples_leaf_high,
                config.min_samples_leaf_type,
            ),
            "min_samples_split": (
                config.min_samples_split_low,
                config.min_samples_split_high,
                config.min_samples_split_type,
            ),
            "n_estimators": (config.n_estimators_low, config.n_estimators_high),
            "criterion": Categorical(config.criterion_categories),
            "max_features": Categorical(config.max_features_categories),
        }

        model = BayesSearchCV(
            estimator=estimator,
            search_spaces=search_space,
            scoring="neg_root_mean_squared_error",
            cv=tss,
            n_jobs=-1,
            n_iter=config.epochs,
            verbose=0,
            refit=True,
        )

        model = model.fit(X_train, y_train)
        dump(
            model,
            config.model_directory
            / f"{config.set_name}_{self.MODEL_NAME.lower()}_{config.future}.pkl",
        )

    def predict(self, model, X_test, y_test, y_scaler, config: RFConfig):
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

        model = load(
            f"{model_directory}/{config.set_name}_{self.MODEL_NAME.lower()}_{config.future}.pkl"
        )
        predictions = model.predict(X_test).reshape(-1, 1)
        predictions = y_scaler.inverse_transform(predictions)
        y_test = y_scaler.inverse_transform(y_test)

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

        print(f"Finished running rf prediction on future window {config.future}")

        metric_outputs = get_metrics(predictions, y_test, 0, self.MODEL_NAME)
        return metric_outputs

    def evaluate(self, X_train, y_train, y_scaler, config: RFConfig):
        """
        _description_

        :param X_train: _description_
        :param y_train: _description_
        :param y_scaler: _description_
        :param config: _description_
        :return: _description_
        """
        time_start = time.time()

        absl.logging.set_verbosity(absl.logging.ERROR)

        self.train(X_train, y_train, None, config)

        print(f"Finished evaluating rf for future {config.future}")

        time_end = time.time()

        return time_end - time_start
