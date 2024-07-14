import pandas as pd
import mlflow
from zenml import step
from typing_extensions import Annotated
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple
from sklearn.base import ClassifierMixin
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

@step(experiment_tracker="mlflow_experiment_tracker")
def model_trainer_random_forest(X_train: pd.DataFrame, y_train: pd.Series, best_parameters: dict) -> Tuple[Annotated[ClassifierMixin, "Random_Forest_Model"], Annotated[float, "Random_Forest_In_Sample_F1_Score"]]:
    """
    Trains the random forest classifier model using the training dataset and the best hyperparameters found during hyperparameter tuning.
    """
    mlflow.sklearn.autolog()
    model = RandomForestClassifier(**best_parameters)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    in_sample_f1_score = f1_score(y_train, y_train_pred, average='weighted')
    return model, in_sample_f1_score
    
