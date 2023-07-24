from basic_nn_config import BasicNNConfig


class ComplexNNConfig(BasicNNConfig):
    regularisation_min: float = 1e-3
    regularisation_max: float = 1e-2
    regularisation_sampling: str = "log"
    l_layer_1_min: int = 1
    l_layer_1_max: int = 100
    l_layer_1_step: int = 10
    neurons_max: int = 5000
