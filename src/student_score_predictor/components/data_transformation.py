import sys
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from student_score_predictor.utils.file_utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        """
        This funciton is responsible for data transformation.
        """
        try:
            numerical_cols = ["writing_score","reading_score"]
            categorical_cols = ["gender",
                                "race_ethnicity",
                                "parental_level_of_education",
                                "lunch",
                                "test_preparation_course"]
            
            numerical_pipeline = Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),
                                                 ("scaler",StandardScaler())])
            
            categorical_pipeline = Pipeline(steps=[
                                  ("imputer",SimpleImputer(strategy="most_frequent")),
                                  ("one_hot_encoder",OneHotEncoder()),
                                  ("scaler",StandardScaler(with_mean=False))])
            
            logging.info("Imputing and Standard scaling of numerical columns completed")
            logging.info("Imputing, one hot encoding and scaling of categorical columns completed")

            preprocessor = ColumnTransformer([
                                    ("num_pipeline",numerical_pipeline,numerical_cols),
                                    ("cat_pipeline",categorical_pipeline, categorical_cols)
                                    ])
            
            return preprocessor
        

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Reading of train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_col = "math_score"
            
            # Dropping target column from both train and test set
            input_feature_train_df = train_df.drop(columns=[target_col], axis=1)
            target_feature_train_df = train_df[target_col]
            
            input_feature_test_df = test_df.drop(columns=[target_col], axis=1)
            target_feature_test_df = test_df[target_col]

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
                )
            
            input_feature_train_df_preprocessed = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_df_preprocessed = preprocessing_obj.transform(input_feature_test_df)
            
            # Columnwise concatenating arrays
            train_arr = np.c_[input_feature_train_df_preprocessed,
                              target_feature_train_df.to_numpy()]
            test_arr = np.c_[input_feature_test_df_preprocessed,
                              target_feature_test_df.to_numpy()]
            
            # Equivalent to np.concatenate((X, y.reshape(-1, 1)), axis=1)
            logging.info("Preprocessing steps completed.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
                )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)

