from models.configuration.i_config import IConfig


class RFConfig(IConfig):
    epd: int
    epochs: int
    max_depth_high: int = 1200
    max_depth_low: int = 10
    min_samples_leaf_high: float = 0.5
    min_samples_leaf_low: float = 0.001
    min_samples_leaf_type: str = "uniform"
    min_samples_split_high: float = 1.0
    min_samples_split_low: float = 0.001
    min_samples_split_type: str = "uniform"
    n_estimators_high: int = 5000
    n_estimators_low: int = 5
    criterion_categories: list[str] = ["squared_error"]
    max_features_categories: list[str] = ["sqrt", "log2", None]
