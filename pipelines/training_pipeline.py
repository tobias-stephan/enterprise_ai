from steps import hp_tuning,model_trainer_random_forest,model_trainer_decision_tree,evaluate_models
from zenml import pipeline
from zenml.client import Client
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

@pipeline(enable_cache=False)
def training_pipeline():
    """ 
    Pipeline to train and deploy a machine learning model using preprocessed and encoded datasets.
    """
    client = Client()
    X_train = client.get_artifact_version("X_train_preprocessed")
    X_val = client.get_artifact_version("X_val_preprocessed")
    X_test = client.get_artifact_version("X_test_preprocessed")
    y_train = client.get_artifact_version("y_train_encoded")
    y_val = client.get_artifact_version("y_val_encoded")
    y_test = client.get_artifact_version("y_test_encoded")
    
    best_rf_params = hp_tuning(X_train, y_train, X_val, y_val, model_type='RandomForest')
    best_dt_params = hp_tuning(X_train, y_train, X_val, y_val, model_type='DecisionTree')
    
    rf_model, rf_in_sample_score = model_trainer_random_forest(X_train, y_train, best_parameters=best_rf_params)
    dt_model, dt_in_sample_score = model_trainer_decision_tree(X_train, y_train, best_parameters=best_dt_params)
    
    deploy, best_model = evaluate_models(rf_model, dt_model, X_test, y_test)
    mlflow_model_deployer_step(model=best_model, deploy_decision=deploy, workers=1)
