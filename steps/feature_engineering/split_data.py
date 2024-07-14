import pandas as pd
from zenml import step
from typing_extensions import Annotated
from sklearn.model_selection import train_test_split
from typing import Tuple


@step
def split_data(dataset:pd.DataFrame, label: str) -> Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_val"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_val"],
    Annotated[pd.Series,"y_test"]]:
    """
    Splits a dataset into training, validation and testing sets, separating features from the target label.
    """
    X = dataset.drop(label,axis=1)
    Y = dataset[label]

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=0)

    return X_train,X_val,X_test,y_train,y_val,y_test