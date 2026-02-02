import os
import sys
import numpy as np
import pandas as pd
import dill
import json
import yaml
from pathlib import Path
from typing import Any
from exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.join("artifacts")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as f:
            return dill.load(f)
    
    except Exception as e:
        raise CustomException(e,sys)
    

def save_json(data: Any, file_path: str, indent: int = 4) -> str:
    """
    Save a Python object as a JSON file.

    Args:
        data: Any JSON-serializable Python object
        file_path: Full file path (e.g., artifacts/metrics.json)
        indent: JSON indentation level

    Returns:
        str: Path of the saved JSON file
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)

        return file_path

    except Exception as e:
        raise CustomException(e, sys)


def read_yaml(file_path: str) -> dict:
    """
    Read a YAML file and return contents as dict.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)