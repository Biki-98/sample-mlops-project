import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,
                             GradientBoostingRegressor,
                             RandomForestRegressor
                             )
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from student_score_predictor.utils.file_utils import save_object, read_yaml, save_models_and_report
from student_score_predictor.utils.file_utils import load_best_model
from student_score_predictor.utils.evaluate_model import evaluate_model, tune_model

params = read_yaml("params.yaml")

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","best_model.dill")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spliting of training and test input data started.")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("Spliting of training and test input data completed successfully.")
            
            models = {
                "random_forest": RandomForestRegressor(),
                "decision_tree": DecisionTreeRegressor(),
                "gradient_boosting": GradientBoostingRegressor(),
                "linear_regression": LinearRegression(),
                "xGBRegressor": XGBRegressor(),
                # "catBoosting_regressor": CatBoostRegressor(verbose=False),
                # "adaBoost_regressor": AdaBoostRegressor(),
            }
            
            logging.info("Starting Training + Prediction + Evaluation of the models.")
            
            raw_model_report: dict = evaluate_model(X_train=X_train, y_train=y_train,
                                                    X_test=X_test, y_test=y_test,
                                                    models=models,
                                                    params=params,
                                                    hyperparameter_tuning=True)
            
            logging.info("Training + Prediction + Evaluation of the models completed successfully.")

            model_report = save_models_and_report(raw_model_report=raw_model_report)
            
            # To get the best model name & score from the model_report
            best_model_name, best_model_info = max(model_report.items(),key=lambda x: x[1]["best_pred_score"])
            best_score = best_model_info["best_pred_score"]
            best_model_path = best_model_info["model_path"]

            logging.info("Best model name and score is obtained successfully.")

            if best_score < 0.6:
                raise CustomException("Best model score is less than 0.6, no best model found")
            
            logging.info(f"Best model {best_model_name} , score: {best_score}")

            save_object(file_path = self.model_trainer_config.trained_model_file_path,
                        obj = best_model_path
                        )
            
            logging.info("Best model is saved in artifacts/best_model.dill")

        except Exception as e:
            raise CustomException(e, sys)

