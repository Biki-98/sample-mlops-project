import os
import sys
import numpy as np
import pandas as pd
import dill
import json
import yaml
from pathlib import Path
from typing import Any, Optional
from exception import CustomException
from logger import logging

def save_object(file_path, obj: Optional[Any] = None):
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
    
def load_best_model(best_model_path: str):
    """
    Handles BOTH cases:
    1. best_model.dill contains a trained model object
    2. best_model.dill contains a string path to the real model
    """

    model_or_path = load_object(best_model_path)

    # Case 1: Pointer file (string path)
    if isinstance(model_or_path, str):
        return load_object(model_or_path)

    # Case 2: Old format (actual model)
    return model_or_path

    
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
    
    
def save_models_and_report(raw_model_report: dict):
    """
    1. Saves the actual model objects to .dill files inside artifacts/models.
    2. Creates a clean dictionary with just paths and scores.
    3. Dumps that clean dictionary to JSON.
    """
    # Save model to artifacts/models/
    dir_path = Path("artifacts/models")
    dir_path.mkdir(parents=True, exist_ok=True)  

    model_report = {}

    for model_name, report in raw_model_report.items():
        # Extract the heavy model object
        model_obj = report["estimator"]

        # Define a path for this specific model
        model_filename = f"{model_name}.dill"
        model_path = os.path.join(dir_path, model_filename)
        
        # Save the model object
        with open(model_path,"wb") as f:
            dill.dump(model_obj, f)
        
        # Construct the json file
        model_report[model_name] = {"best_pred_score": report["best_pred_score"],
                                    "best_params": report["best_params"],
                                    "train_score": report["train_score"],
                                    "model_path": model_path  
                                    }
        
        json_report_path = save_json(data=model_report,
                                     file_path="artifacts/model_report.json")

    logging.info(f"Json report saved to {json_report_path}")
    logging.info(f"Models saved to {dir_path}")

    return model_report






