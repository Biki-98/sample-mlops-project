import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

# Hyperparameter tuning
def tune_model(model, param_grid, X_train, y_train):
    """
    This function is for hyperparameter tuning.
    """
    rs = RandomizedSearchCV(estimator=model,
                            param_distributions=param_grid,
                            cv=3,
                            n_iter=10,
                            n_jobs=-1,
                            refit=True)
    
    rs.fit(X_train, y_train)
    return rs.best_estimator_, rs.best_params_

def evaluate_model(X_train,y_train,X_test,y_test,models,params,hyperparameter_tuning=True):
    """
    This function is for evaluating model performance with hyperparameter tuning and for single
    algorithm/model also.
    """
    try:
        raw_model_report = {}
        for model_name, model in models.items():
            if hyperparameter_tuning:
                logging.info(f"Hyperparameter tuning of {model_name} has started.")
                
                estimator, best_params = tune_model(model=model,
                                                    param_grid=params[model_name],
                                                    X_train=X_train,
                                                    y_train=y_train)
                
                logging.info(f"Hyperparameter tuning of {model_name} has completed successfully.")

            else:
                estimator = model
                best_params = None
                
                logging.info(f"Training of the {model_name} has started.")
                
                model.fit(X_train, y_train)
                
                logging.info(f"Training of the {model_name} is completed successfully.")

            logging.info(f"Prediction of the {model_name} has started.")
            
            y_train_pred = estimator.predict(X_train)
            y_test_pred = estimator.predict(X_test)
            
            logging.info(f"Prediction of the {model_name} is completed successfully.")

            logging.info(f"Evaluation of {model_name} based on R2-score is started.")
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            logging.info(f"Evaluation of {model_name} based on R2-score is completed successfully.")

            raw_model_report[model_name] = {"estimator": estimator,
                                            "best_pred_score": test_model_score,
                                            "best_params": best_params,
                                            "train_score": train_model_score
                                            }

        return raw_model_report

    except Exception as e:
         raise CustomException(e, sys)































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