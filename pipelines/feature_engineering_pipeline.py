from steps import loading_data,data_cleaning,split_data,label_encoding,create_preprocessing_pipeline,feature_preprocessor
from zenml import pipeline

@pipeline
def feature_engineering_pipeline():
    """"
    Pipeline function for performing feature engineering on patient data.
    """
    dataset = loading_data("./data/app_data.xlsx")
    cleaned_dataset = data_cleaning(dataset)
    X_train,X_val,X_test,y_train,y_val,y_test = split_data(cleaned_dataset,"Diagnosis")
    pipeline = create_preprocessing_pipeline(cleaned_dataset,"Diagnosis")
    X_train,X_val,X_test,pipeline = feature_preprocessor(pipeline,X_train,X_val,X_test)
    y_train,y_val,y_test = label_encoding(y_train,y_val,y_test)