import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import ClassifierMixin
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import mlflow

@step(experiment_tracker="mlflow_experiment_tracker")
def evaluate_models(rf_model: ClassifierMixin, dt_model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[Annotated[bool,"deployment_decision"], Annotated[ClassifierMixin,"deployed_model"]]:
    """
    Evaluates the trained model and returns a deployment decision based on the f1 score.
    """

    rf_prediction = rf_model.predict(X_test)
    rf_score = rf_model.score(X_test,y_test)
    rf_recall = recall_score(y_test, rf_prediction)
    rf_precision = precision_score(y_test, rf_prediction)
    rf_f1 = f1_score(y_test, rf_prediction)
    mlflow.log_metric("RandomForest_Accuracy", rf_score)
    mlflow.log_metric("RandomForest_Recall", rf_recall)
    mlflow.log_metric("RandomForest_Precision", rf_precision)
    mlflow.log_metric("RandomForest_F1", rf_f1)


    dt_prediction = dt_model.predict(X_test)
    dt_score = dt_model.score(X_test,y_test)
    dt_recall = recall_score(y_test, dt_prediction)
    dt_precision = precision_score(y_test, dt_prediction)
    dt_f1 = f1_score(y_test, dt_prediction)
    mlflow.log_metric("DecisionTree_Accuracy", dt_score)
    mlflow.log_metric("DecisionTree_Recall", dt_recall)
    mlflow.log_metric("DecisionTree_Precision", dt_precision)
    mlflow.log_metric("DecisionTree_F1", dt_f1)    

  
    if rf_f1 > dt_f1:
        best_model = rf_model
        deploy = True if rf_f1 > 0.8 else False
        print("Deployed model with a higher f1 score: Random Forest Classifier Model") if rf_f1 > 0.8 else print("No model is deployed")
    else:
        best_model = dt_model
        deploy = True if dt_f1 > 0.8 else False
        print("Deployed model with a higher f1 score: Decision Tree Classifier Model") if dt_f1 > 0.8 else print("No model is deployed")
    return deploy, best_model
