import sys
import os
import pandas as pd
from logger import logging
from exception import CustomException
from student_score_predictor.utils.file_utils import load_best_model,load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","best_model.dill")
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            
            logging.info("best_model.dill and preprocessor.pkl file paths are saved.")
            
            model=load_best_model(best_model_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            logging.info("Starting prediction data preprocessing and prediction.")
            
            data_preprocessed = preprocessor.transform(features)
            preds = model.predict(data_preprocessed)
            
            logging.info("Finished prediction data preprocessing and prediction.")
            
            return preds
            
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
