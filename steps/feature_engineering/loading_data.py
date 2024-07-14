import pandas as pd
from zenml import step
from typing_extensions import Annotated

@step(enable_cache=False)
def loading_data(filename: str) -> Annotated[pd.DataFrame,"input_data"]:
    """
    Loads data from a cohort of pediatric patients with suspected appendicitis admitted with abdominal pain to Children’s Hospital St. Hedwig in Regensburg, Germany, between 2016 and 2021.
    
    Parameters:
    filename (str): Path to the Excel file.

    Returns:
    Annotated[pd.DataFrame, "input_data"]: DataFrame with patient data.
    """
    data = pd.read_excel(filename)

    return data