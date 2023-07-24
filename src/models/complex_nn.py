import keras_tuner as kt
import numpy as np
from keras import regularizers
from keras.layers import LSTM, Dense, Dropout, Flatten, InputLayer
from keras.losses import MeanSquaredError
from keras.models import Sequential
from keras.optimizers import Adam

from models.configuration.complex_nn_config import ComplexNNConfig
from models.basic_nn import BasicNNModel


class ComplexNNModel(BasicNNModel):
    """
    _description_
    """

    MODEL_NAME = "Complex_nn"

    def kt_model(hp: kt.HyperParameters, config: ComplexNNConfig) -> Sequential:
        """
        _description_

        :param hp: _description_
        :param config: _description_
        :return: _description_
        """
        X = np.load("X_train_3d.npy")

        hp_activation = hp.Choice("activation", values=["relu", "tanh"])
        hp_learning_rate = hp.Float(
            "lr",
            min_value=config.learning_rate_min,
            max_value=config.learning_rate_max,
            sampling=config.learning_rate_sampling,
        )
        hp_reg = hp.Float(
            "reg",
            min_value=config.regularisation_min,
            max_value=config.regularisation_max,
            sampling=config.regularisation_sampling,
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

        hp_l_layer_1 = hp.Int(
            "l_layer_1",
            min_value=config.l_layer_1_min,
            max_value=config.l_layer_1_max,
            step=config.l_layer_1_step,
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
        model.add(InputLayer((X.shape[1], X.shape[2])))
        model.add(
            LSTM(
                hp_l_layer_1,
                return_sequences=True,
                activity_regularizer=regularizers.l1(hp_reg),
            )
        )
        model.add(Dropout(hp_dropout))
        model.add(Flatten())

        while neuron_count > 20 and layers < 20:
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
