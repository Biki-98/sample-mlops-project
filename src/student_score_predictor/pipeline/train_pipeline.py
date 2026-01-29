import sys
from ..components.data_ingestion import DataIngestionConfig, DataIngestion
from ..components.data_transformation import DataTransformationConfig, DataTransformation
from ..components.model_trainer import ModelTrainerConfig, ModelTrainer
from src.exception import CustomException
from src.logger import logging

def run_pipeline():
    try:
        logging.info("---Staring Pipeline---")
        # start data ingestion
        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.start_data_ingestion()
    
        # start preprocessing/transformation
        data_transformation = DataTransformation()
        # train_transformed, test_transformed, prep_obj_path  = (data_transformation.
                                                # initiate_data_transformation(train_data,test_data))
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data,
                                                                                  test_data)
        
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr)
        logging.info("---Pipeline executed successfully---")
    
    except Exception as e:
        raise CustomException(e,sys)


if __name__=="__main__":
    run_pipeline()
    