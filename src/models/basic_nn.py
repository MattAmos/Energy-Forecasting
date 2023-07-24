import os
from time import time

import absl.logging
import keras_tuner as kt
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.losses import MeanSquaredError
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

from models.configuration.basic_nn_config import BasicNNConfig
from models.i_model import IModel
from performance_analysis import cross_val_metrics, get_metrics, make_csvs


class BasicNNModel(IModel):
    """
    _description_
    """

    MODEL_NAME = "Basic_nn"

    @staticmethod
    def kt_model(hp: kt.HyperParameters, config: BasicNNConfig) -> Sequential:
        """
        _description_

        :param hp: _description_
        :param config: _description_
        :return: _description_
        """
        hp_activation = hp.Choice("activation", values=config.activation)
        hp_learning_rate = hp.Float(
            "lr",
            min_value=config.learning_rate_min,
            max_value=config.learning_rate_max,
            sampling=config.learning_rate_sampling,
        )
        hp_dropout = hp.Float(
            "dropout",
            min_value=config.dropout_min,
            max_value=config.dropout_max,
            sampling=config.dropout_sampling,
        )
        hp_neuron_pct = hp.Float(
            "NeuronPct",
            min_value=config.neuron_pct_min,
            max_value=config.neuron_pct_max,
            sampling=config.neuron_pct_sampling,
        )
        hp_neuron_shrink = hp.Float(
            "NeuronShrink",
            min_value=config.neuron_shrink_min,
            max_value=config.neuron_shrink_max,
            sampling=config.neuron_shrink_sampling,
        )
        hp_max_neurons = hp.Int(
            "neurons",
            min_value=config.neurons_min,
            max_value=config.neurons_max,
            step=config.neurons_step,
        )

        neuron_count = int(hp_neuron_pct * hp_max_neurons)
        layers = 0

        model = Sequential()

        while neuron_count > 5 and layers < 5:
            model.add(Dense(units=neuron_count, activation=hp_activation))
            model.add(Dropout(hp_dropout))
            layers += 1
            neuron_count = int(neuron_count * hp_neuron_shrink)

        model.add(Dense(1, "linear"))

        model.compile(
            loss=MeanSquaredError(),
            optimizer=Adam(learning_rate=hp_learning_rate),
            metrics=[
                "mean_squared_error",
                "mean_absolute_error",
                "mean_absolute_percentage_error",
            ],
        )

        return model

    def train(
        self,
        X_train,
        y_train,
        y_scaler,
        config: BasicNNConfig,
    ):
        """
        Train the model

        :param X_train: _description_
        :param y_train: _description_
        :param y_scaler: _description_
        :param config: _description_
        """
        tuner = kt.Hyperband(
            BasicNNModel.kt_model,
            objective="mean_absolute_percentage_error",
            max_epochs=config.epochs,
            factor=3,
            directory=config.model_directory / (config.set_name + "_kt_dir"),
            project_name="kt_model_" + str(config.future),
            overwrite=True,
        )

        monitor = EarlyStopping(
            monitor="mean_absolute_percentage_error",
            min_delta=1,
            patience=5,
            verbose=0,
            mode="auto",
            restore_best_weights=True,
        )

        tuner.search(
            X_train,
            y_train,
            verbose=0,
            epochs=config.epochs,
            validation_split=0.2,
            batch_size=config.batch_size,
            callbacks=[monitor],
        )

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(best_hps)

        # Split on a 3 monthly basis
        tss = TimeSeriesSplit(n_splits=10, test_size=config.epd * 90, gap=0)
        fold = 0
        total_metrics = {}

        for train_idx, val_idx in tss.split(X_train, y_train):
            fold_name = "Fold_" + str(fold)
            X_t = X_train[train_idx]
            X_v = X_train[val_idx]
            y_t = y_train[train_idx]
            y_v = y_train[val_idx]

            if fold == 9:
                history = model.fit(
                    X_t,
                    y_t,
                    verbose=0,
                    epochs=config.epochs,
                    callbacks=[monitor],
                    batch_size=config.batch_size,
                    validation_data=(X_v, y_v),
                )
                graphs_directory = os.getcwd() + "/graphs"
                self.save_plots(
                    history, graphs_directory, config.set_name, config.future
                )
                model.save(
                    config.model_directory
                    / (f"{config.set_name}_{self.MODEL_NAME.lower()}_{config.future}")
                )

            model.fit(
                X_t,
                y_t,
                verbose=0,
                epochs=config.epochs,
                callbacks=[monitor],
                batch_size=config.batch_size,
            )
            preds = model.predict(X_v, verbose=0)
            preds = y_scaler.inverse_transform(preds)
            metrics = get_metrics(preds, y_v, 1, self.MODEL_NAME)
            total_metrics[fold_name] = metrics

            fold += 1

        cross_val_metrics(
            total_metrics,
            config.set_name,
            config.future,
            self.MODEL_NAME,
        )

    def predict(self, model, X_test, y_test, y_scaler, config: BasicNNConfig):
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
            f"{model_directory}/{config.set_name}_{self.MODEL_NAME.lower()}_{config.future}"
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
            "Finished running basic prediction on future window {0}".format(
                config.future
            )
        )

        metric_outputs = get_metrics(predictions, y_test, 0, self.MODEL_NAME)
        return metric_outputs

    def evaluate(self, X_train, y_train, y_scaler, config: BasicNNConfig):
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
        tf.compat.v1.logging.set_verbosity(30)

        self.train(X_train, y_train, y_scaler, config)

        print(f"Finished evaluating basic nn for future {config.future}")

        time_end = time.time()

        return time_end - time_start

    def save_plots(self, history, config: BasicNNConfig):
        """
        _description_

        :param history: _description_
        :type history: _type_
        :param graphs_directory: _description_
        :type graphs_directory: _type_
        :param set_name: _description_
        :type set_name: _type_
        :param future: _description_
        :type future: _type_
        """
        graph_names = {
            "Loss": "loss",
            "MAE": "mean_absolute_error",
            "MSE": "mean_squared_error",
            "MAPE": "mean_absolute_percentage_error",
        }

        for name, value in graph_names.items():
            graph_loc = f"{config.graphs_directory}/{config.set_name}_{self.MODEL_NAME}_{config.future}_{name}.png"
            if os.path.exists(graph_loc):
                os.remove(graph_loc)

            val_name = "val_" + value
            plt.plot(history.history[value])
            plt.plot(history.history[val_name])
            plt.title(f"Basic NN {name} for {config.set_name} {config.future}")
            plt.ylabel(name)
            plt.xlabel("epoch")
            plt.legend(["train", "test"], loc="upper left")
            plt.savefig(graph_loc)
