import pandas as pd
from zenml import step
from typing_extensions import Annotated

@step(enable_cache=False)
def inference_data_loader(filename: str) -> Annotated[pd.DataFrame,"input_data"]:
    """
    Loads a Excel File and transforms it to a Pandas DataFrame
    """
    data = pd.read_excel(filename)

    columns_to_drop = ['Diagnosis', 'Length_of_Stay' , 'Management' , 'Severity' , 'Diagnosis_Presumptive' , 'Segmented_Neutrophils' , 'Appendix_Wall_Layers' , 'Target_Sign' , 'Appendicolith' , 'Perfusion' , 'Perforation' , 'Surrounding_Tissue_Reaction' , 'Appendicular_Abscess' , 'Abscess_Location' , 'Pathological_Lymph_Nodes' , 'Lymph_Nodes_Location' , 'Bowel_Wall_Thickening' , 'Conglomerate_of_Bowel_Loops' , 'Ileus' , 'Coprostasis' , 'Meteorism' , 'Enteritis' , 'Gynecological_Findings']
    data = data.drop(columns=columns_to_drop)

    return data