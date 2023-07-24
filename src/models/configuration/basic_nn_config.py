from i_config import IConfig


class BasicNNConfig(IConfig):
    batch_size: int
    epochs: int
    epd: int
    activation: list[str] = ["relu", "tanh"]
    learning_rate_min: float = 1e-4
    learning_rate_max: float = 1e-2
    learning_rate_sampling: str = "log"
    dropout_min: float = 1e-3
    dropout_max: float = 0.5
    dropout_sampling: str = "linear"
    neuron_pct_min: float = 1e-3
    neuron_pct_max: float = 1.0
    neuron_pct_sampling: str = "linear"
    neuron_shrink_min: float = 1e-3
    neuron_shrink_max: float = 1.0
    neuron_shrink_sampling: str = "linear"
    neurons_min: int = 10
    neurons_max: int = 200
    neurons_step: int = 10
