import absl.logging
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt import dump, load

import time
import xgboost as xgb

from performance_analysis import make_csvs, get_metrics

from models.i_model import IModel
from models.configuration.xgb_config import XGBConfig


class XGBoostModel(IModel):
    MODEL_NAME = "XGBoost"

    def train(self, X_train, y_train, _, config: XGBConfig):
        """
        _description_

        :param X_train: _description_
        :param y_train: _description_
        :param _: _description_
        :param config: _description_
        """
        length = X_train.shape[0]
        X_train_temp = X_train[: int(length * config.split), :]
        y_train_temp = y_train[: int(length * config.split), :]
        X_val = X_train[int(length * config.split) :, :]
        y_val = y_train[int(length * config.split) :, :]

        tss = TimeSeriesSplit(n_splits=5, test_size=config.epd * 90, gap=0)
        estimator = xgb.XGBRegressor(
            booster="gbtree",
            early_stopping_rounds=50,
            objective="reg:squarederror",
            verbosity=0,
        )

        search_space = {
            "learning_rate": (
                config.learning_rate_low,
                config.learning_rate_high,
                config.learning_rate_type,
            ),
            "min_child_weight": (
                config.min_child_weight_low,
                config.min_child_weight_high,
            ),
            "max_depth": (config.max_depth_low, config.max_depth_high),
            "subsample": (
                config.subsample_low,
                config.subsample_high,
                config.subsample_type,
            ),
            "colsample_bytree": (
                config.colsample_bytree_low,
                config.colsample_bytree_high,
                config.colsample_bytree_type,
            ),
            "reg_lambda": (
                config.reg_lambda_low,
                config.reg_lambda_high,
                config.reg_lambda_type,
            ),
            "reg_alpha": (
                config.reg_alpha_low,
                config.reg_alpha_high,
                config.reg_alpha_type,
            ),
            "gamma": (config.gamma_low, config.gamma_high, config.gamma_type),
            "n_estimators": (config.n_estimators_low, config.n_estimators_high),
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
        model = model.fit(
            X_train_temp, y_train_temp, eval_set=[(X_val, y_val)], verbose=False
        )
        dump(
            model,
            config.model_directory
            / f"{config.set_name}_{self.MODEL_NAME}_{config.future}.pkl",
        )

    def predict(self, model, X_test, y_test, y_scaler, config: XGBConfig):
        """
        _description_

        :param model: _description_
        :param X_test: _description_
        :param y_test: _description_
        :param y_scaler: _description_
        :param config: _description_
        :return: _description_
        """
        model = load(
            config.model_directory
            / f"{config.set_name}_{self.MODEL_NAME}_{config.future}.pkl"
        )
        predictions = model.predict(X_test).reshape(-1, 1)
        predictions = y_scaler.inverse_transform(predictions)
        y_test = y_scaler.inverse_transform(y_test)

        # MA: Not sure if this is right, but could use something like this to get the dates
        pred_dates_test = X_test.index.dt.strftime("%Y-%m-%d").values

        make_csvs(
            config.csv_directory,
            predictions,
            y_test,
            pred_dates_test,
            config.set_name,
            config.future,
            self.MODEL_NAME,
        )

        print(f"Finished running xgb prediction on future window {config.future}")

        metric_outputs = get_metrics(predictions, y_test, 0, self.MODEL_NAME)
        return metric_outputs

    def evaluate(self, X_train, y_train, y_scaler, config: XGBConfig):
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

        self.train(X_train, y_train, config)

        print(f"Finished evaluating xgb for future {config.future}")

        time_end = time.time()
        return time_end - time_start
