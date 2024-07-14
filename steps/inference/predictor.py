import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    input_data: pd.DataFrame,
    output_file: str = "predictions.csv"
) -> np.ndarray:

    """Run an inference request against a prediction service"""
    service.start(timeout=10) 
    prediction = service.predict(input_data)

    # Save the predictions to a CSV file
    prediction_df = pd.DataFrame(prediction, columns=["prediction"])
    prediction_df.to_csv(output_file, index=False)

    return prediction