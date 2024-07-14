from zenml import step
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import optuna
import pandas as pd
from functools import partial
from typing_extensions import Annotated
from sklearn.tree import DecisionTreeClassifier

def objective(trial, X_train, y_train, X_val, y_val, model_type='RandomForest'):
    """ 
    This function defines the objective for the Optuna optimization.
    """
    
    if model_type == 'RandomForest':
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 1, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=42
        )
    else:  # DecisionTree
        max_depth = trial.suggest_int('max_depth', 1, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=42
        )
        
    model.fit(X_train, y_train)
    f1 = f1_score(y_val, model.predict(X_val))
    return f1

@step
def hp_tuning(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, model_type: str = 'RandomForest', trials: int = 100) -> Annotated[dict, "Best hyperparameters"]:
    """
    This step tunes the hyperparameters for both models using Optuna.
    """
    
    obj = partial(objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, model_type=model_type)
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=trials)
    best_params = study.best_params
    return best_params

