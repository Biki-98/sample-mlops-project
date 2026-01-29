import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def evaluate_model(X_train,y_train,X_test,y_test,models):
    """
    This function is for evaluating model performance.
    """
    try:
        # To store model performance for each model
        model_report = {}
        
        for model_name, model in models.items():
            # Traning the model
            logging.info(f"Training of the {model_name} has started.")
            model.fit(X_train,y_train)
            
            logging.info(f"Training of the {model_name} is completed successfully.")
            
            # Prediction of the model
            logging.info(f"Prediciton of the {model_name} has started.")
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            logging.info(f"Prediction of the {model_name} is completed successfully.")

            # Evaluating model performance based on R2-score
            logging.info(f"Evaluation of {model_name} based on R2-score is started.")
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            logging.info(f"Evaluation of {model_name} based on R2-score is completed successfully.")

            model_report[model_name] = test_model_score

        return model_report
    
    except Exception as e:
        raise CustomException(e,sys)



































# def evaluate_models(X_train, y_train,X_test,y_test,models):
#     try:
#         report = {}

#         for i in range(len(list(models))):

#             # model = list(models.values())[i]
#             # para=param[list(models.keys())[i]]

#             # gs = GridSearchCV(model,para,cv=3)
#             # gs.fit(X_train,y_train)

#             # model.set_params(**gs.best_params_)
#             model.fit(X_train,y_train)

#             #model.fit(X_train, y_train)  # Train model

#             y_train_pred = model.predict(X_train)

#             y_test_pred = model.predict(X_test)

#             train_model_score = r2_score(y_train, y_train_pred)

#             test_model_score = r2_score(y_test, y_test_pred)

#             report[list(models.keys())[i]] = test_model_score

#         return report

#     except Exception as e:
#         raise CustomException(e, sys)