from pydantic import BaseModel
from pathlib import Path
from abc import ABC


class IConfig(ABC, BaseModel):
    """
    Interface for configuration objects
    """

    model_directory: Path
    csv_directory: Path
    graphs_directory: Path
    set_name: str
    future: str
