import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated

@step(enable_cache=False)
def data_cleaning(dataset:pd.DataFrame) -> Annotated[pd.DataFrame,"cleaned_data"]:
    """
    Cleans the data by removing empty rows from the "Diagnosis" target variable. 
    Free text fields and the two target variables "Management" and "Severity" as well as the data collected on discharge are removed.
    Features that are missing more than 300 values (approx. 60%) are removed.

    Parameters:
    dataset:pd.DataFrame: Loaded DataFrame.

    Returns:
    Annotated[pd.DataFrame, "cleaned_data"]: Cleaned DataFrame with patient data.
    """
    
    cleaned_data = dataset.dropna(subset=["Diagnosis"])
    columns_to_drop = ["Severity", "Management", "Length_of_Stay", "Lymph_Nodes_Location", "Abscess_Location", "Gynecological_Findings", "Diagnosis_Presumptive"]
    cleaned_data = cleaned_data.drop(columns=columns_to_drop)

    missing_values = cleaned_data.isnull().sum()
    nan_threshold = 300
    empty_values = missing_values[missing_values > nan_threshold]
    
    column_values = empty_values.index.tolist()
    cleaned_data = cleaned_data.drop(columns=column_values)

    return cleaned_data