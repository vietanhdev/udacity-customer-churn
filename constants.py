"""Constants for Churn Predictor library
   Author: Viet-Anh Nguyen
   Created at: 24/01/2022
"""

CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

DATA_PREPARATION_KEEP_COLUMNS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio',
    'Gender_Churn',
    'Education_Level_Churn',
    'Marital_Status_Churn',
    'Income_Category_Churn',
    'Card_Category_Churn'
]

DEFAULT_DATA_PATH = "data/bank_data.csv"
DEFAULT_EDA_RESULT_PATH = "eda"
DEFAULT_MODELS_PATH = "models"
DEFAULT_EVALUATION_PATH = "results"

DEFAULT_PARAMS_GRID = {
    'n_estimators': [200],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5],
    'criterion': ['gini', 'entropy']
}
