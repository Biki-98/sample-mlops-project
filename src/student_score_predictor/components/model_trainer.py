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
from student_score_predictor.utils.file_utils import save_object, save_json
from student_score_predictor.utils.evaluate_model import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","best_model.pkl")

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
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            logging.info("Starting Training + Prediction + Evaluation of the models-->")
            
            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train,
                                                X_test=X_test, y_test=y_test,
                                                models=models)
            logging.info("Training + Prediction + Evaluation of the models completed successfully.")

            # To get the best model score from the model_report
            best_model_name, best_model_score = max(model_report.items(), key=lambda x: x[1])

            logging.info("Best model name and score is obtained successfully.")

            if best_model_score < 0.6:
                raise CustomException("Best model score is less than 0.6, no best model found")
            
            logging.info(f"Best model {best_model_name} , score: {best_model_score}")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model_name
            )
            
            # Saving the model report as a json file in artifacts
            save_json(model_report, "artifacts/model_report.json")
            
            logging.info("Model report is saved inside artifacts")
            
            return f"Best model {best_model_name} , score: {best_model_score}"


        except Exception as e:
            raise CustomException(e, sys)

