from models.configuration.i_config import IConfig


class XGBConfig(IConfig):
    epd: int = 10
    epochs: int = 100
    split: float = 0.9
    learning_rate_high: float = 1.0
    learning_rate_low: float = 0.01
    learning_rate_type: str = "log-uniform"
    min_child_weight_high: int = 10
    min_child_weight_low: int = 0
    max_depth_high: int = 50
    max_depth_low: int = 1
    subsample_high: float = 1.0
    subsample_low: float = 0.01
    subsample_type: str = "uniform"
    colsample_bytree_high: float = 1.0
    colsample_bytree_low: float = 0.01
    colsample_bytree_type: str = "log-uniform"
    reg_lambda_high: float = 1.0
    reg_lambda_low: float = 1e-9
    reg_lambda_type: str = "log-uniform"
    reg_alpha_high: float = 1.0
    reg_alpha_low: float = 1e-9
    reg_alpha_type: str = "log-uniform"
    gamma_high: float = 0.5
    gamma_low: float = 1e-9
    gamma_type: str = "log-uniform"
    n_estimators_high: int = 5000
    n_estimators_low: int = 5
