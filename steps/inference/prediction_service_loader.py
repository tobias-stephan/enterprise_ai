from zenml import step
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService

@step(enable_cache=False)
def prediction_service_loader(pipeline_name: str, step_name: str) -> MLFlowDeploymentService:
    """
    Finds and returns the MLflow prediction service deployed by the specified pipeline step.
    """
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = model_deployer.find_model_server(pipeline_name=pipeline_name,
        pipeline_step_name=step_name,)
    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{step_name} step in the {pipeline_name} "
        )

    return existing_services[0]    