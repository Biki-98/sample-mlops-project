import os
import sys
from exception import CustomException
from logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts","data.csv")
    train_data_path: str = os.path.join("artifacts","train.csv")
    test_data_path: str = os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def start_data_ingestion(self):
        logging.info("Entering the data ingestion component")
        try:
            df = pd.read_json("research\data\stud.csv")
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save raw snapshot in same format
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # start train test split
            logging.info("train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save raw snapshot of train and test sets in same format
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            train_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
                    )


        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    obj = DataIngestion()
    obj.start_data_ingestion()




# # Other way to do the above
# class DataIngestion:
#     def initiate_data_ingestion(self):
#         logging.info("Starting data ingestion")

#         df = pd.read_csv("notebook/data/stud.csv")

#         os.makedirs("artifacts", exist_ok=True)

#         # Save raw snapshot
#         df.to_csv("artifacts/raw.csv", index=False)

#         train_df, test_df = train_test_split(
#             df, test_size=0.2, random_state=42
#         )

#         train_df.to_csv("artifacts/train.csv", index=False)
#         test_df.to_csv("artifacts/test.csv", index=False)

#         logging.info("Data ingestion completed")

#         return (
#             "artifacts/train.csv",
#             "artifacts/test.csv"
#         )
