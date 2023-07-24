from models.configuration.i_config import IConfig


class BaselineRegressorConfig(IConfig):
    epochs: int
    batch_size: int
