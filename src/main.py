import math
import os

import numpy as np

from datacleaner import (
    create_dataset_2d,
    create_dataset_3d,
    data_cleaning_pipeline,
    feature_adder,
    finalise_data,
    load_datasets,
)
from models.baseline_regressor import BaselineRegressor
from models.basic_nn import BasicNNModel
from models.complex_nn import ComplexNNModel
from models.configuration.baseline_regressor_config import BaselineRegressorConfig
from models.configuration.basic_nn_config import BasicNNConfig
from models.configuration.complex_nn_config import ComplexNNConfig
from models.configuration.rf_config import RFConfig
from models.configuration.xgb_config import XGBConfig
from models.rf_model import RFModel
from models.xgboost_model import XGBoostModel
from pathlib import Path

from performance_analysis import make_metrics_csvs, normalise_metrics


def get_pipelines(
    project_path: Path,
    set_name: str,
    epd: int,
    future: int,
    epochs: int,
    batch_size: int,
):
    kwargs = {
        "csv_directory": project_path / "csvs",
        "model_directory": project_path / "models",
        "graphs_directory": project_path / "graphs",
        "set_name": set_name,
        "epd": epd,
        "future": future,
        "epochs": epochs,
    }
    return {
        BasicNNModel.MODEL_NAME: {
            "x_train": X_train_2d,
            "y_train": y_train,
            "x_test": X_test_2d,
            "y_test": y_test,
            "model": BasicNNModel,
            "config": BasicNNConfig(batch_size=batch_size, **kwargs),
            "metrics": {},
        },
        ComplexNNModel.MODEL_NAME: {
            "x_train": X_train_3d,
            "y_train": y_train,
            "x_test": X_test_3d,
            "y_test": y_test,
            "model": ComplexNNModel,
            "config": ComplexNNConfig(batch_size=batch_size, *kwargs),
            "metrics": {},
        },
        XGBoostModel.MODEL_NAME: {
            "x_train": X_train_2d,
            "y_train": y_train,
            "x_test": X_test_2d,
            "y_test": y_test,
            "model": XGBoostModel,
            "config": XGBConfig(**kwargs),
        },
        RFModel.MODEL_NAME: {
            "x_train": X_train_2d,
            "y_train": y_train.reshape(-1),
            "x_test": X_test_2d,
            "y_test": y_test,
            "model": RFModel,
            "config": RFConfig(**kwargs),
            "metrics": {},
        },
        BaselineRegressor.MODEL_NAME: {
            "x_train": X_train_2d,
            "y_train": y_train,
            "x_test": X_test_2d,
            "y_test": y_test,
            "config": BaselineRegressorConfig(batch_size=batch_size, **kwargs),
            "metrics": {},
        },
    }


if __name__ == "__main__":
    project_path = Path(__file__).parent / ".."
    csv_directory = project_path / "csvs"

    # These are the values that need changing per different dataset
    file_path = csv_directory + "/matlab_temp.xlsx"
    set_name = "matlab"
    target = "SYSLoad"
    trend_type = "Additive"
    epd = 48
    future = 0

    cleaning = 0
    training = 1
    predicting = 1
    eval_tpot = 0

    partition = 5000
    data_epochs = 10

    # Don't know what else to put in cleaning parameters tbh
    # put more research into this area
    cleaning_parameters = {
        # This is seriously jank. It works for now, but golly...
        "pca_dimensions": [None, math.inf, -math.inf],
        "scalers": ["standard", "minmax"],
    }

    window = 10
    split = 0.8
    epochs = 1
    batch_size = 32

    if cleaning:
        data, outputs = feature_adder(
            csv_directory, file_path, target, trend_type, future, epd, set_name
        )

        # Decide on exactly what size this partition should be
        # Essentially this grid search is shit and has to be optimized later on down the track
        # It'll just get the job done for now
        best_results = data_cleaning_pipeline(
            data[:partition],
            outputs[:partition],
            cleaning_parameters,
            target,
            split,
            data_epochs,
            batch_size,
            csv_directory,
        )

    else:
        if (
            os.path.exists(csv_directory + "/best_data_parameters.csv")
            and os.path.exists(
                csv_directory + "/" + set_name + "_data_" + str(future) + ".csv"
            )
            and os.path.exists(
                csv_directory + "/" + set_name + "_outputs_" + str(future) + ".csv"
            )
        ):
            best_results = pd.read_csv(
                csv_directory + "/best_data_parameters.csv"
            ).to_dict("index")
            best_results = best_results.get(0)
            data, outputs = load_datasets(csv_directory, set_name, future)

        else:
            data, outputs = feature_adder(
                csv_directory, file_path, target, trend_type, future, epd, set_name
            )

            # Decide on exactly what size this partition should be
            # Essentially this grid search is shit and has to be optimized later on down the track
            # It'll just get the job done for now
            best_results = data_cleaning_pipeline(
                data[:partition],
                outputs[:partition],
                cleaning_parameters,
                target,
                split,
                data_epochs,
                batch_size,
                csv_directory,
            )

    X_frame, y_data, pred_dates, y_scaler = finalise_data(
        data, outputs, target, best_results
    )
    length = X_frame.shape[0]

    pred_dates_test = pred_dates[int(length * split) + window :]

    y_test = y_data[int(length * split) + window :]
    X_test_2d = create_dataset_2d(X_frame[int(length * split) :], window)
    X_test_3d = create_dataset_3d(X_frame[int(length * split) :], window)

    y_train = y_data[window : int(length * split)]
    X_train_2d = create_dataset_2d(X_frame[: int(length * split)], window)
    X_train_3d = create_dataset_3d(X_frame[: int(length * split)], window)

    pipelines = get_pipelines(
        project_path, set_name, epd, future, data_epochs, batch_size
    )

    if training:
        np.save("X_train_3d.npy", X_train_3d)
        for label, params in pipelines.items():
            model = params["model"]
            pipelines[model]["metrics"]["TIME"] = model.evaluate(
                X_train=params["x_train"],
                y_train=params["y_train"],
                y_scaler=y_scaler,
                config=params["config"],
            )

    if predicting:
        for label, params in pipelines.items():
            model = params["model"]
            pipelines[model]["metrics"].update(
                model.predict(
                    X_test=params["x_test"],
                    y_test=params["y_test"],
                    y_scaler=y_scaler,
                    config=params["config"],
                )
            )

        metrics = [x["metrics"] for x in pipelines.values()]
        metrics = normalise_metrics(metrics, training)
        metrics = {k: metrics[ii] for ii, k in enumerate(pipelines.keys())}

        make_metrics_csvs(csv_directory, metrics, set_name, future, training)

    if eval_tpot:
        # Problem still exists with tpot for whatever reason
        tpot_evaluate(
            future,
            set_name,
            X_train_2d,
            y_train.reshape(-1),
            pred_dates_test,
            X_test_2d,
            y_test.reshape(-1),
            y_scaler,
        )

    if os.path.exists("X_train_3d.npy"):
        os.remove("X_train_3d.npy")
