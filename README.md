# Predict Customer Churn

Course project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity.

Code in `churn_library.py` completes the process for solving the data science process including:
- EDA
- Feature Engineering (including encoding of categorical variables)
- Model Training
- Prediction
- Model Evaluation

## Environment Setup

- Requirement: Python 3.x

```
pip install requirements.txt
```

## Training & Evaluation

Example running command:

```
python churn_library.py --data="data/bank_data.csv" \
                        --log_path="churn_library.log" \
                        --eda_output="eda" \
                        --model_output="models" \
                        --evaluation_output="results" \
                        --feature_importance_output="feature_importance.png" \
                        --test_size=0.3
```

Adjust parameters for grid search (DEFAULT_PARAMS_GRID) in `constants.py`.

## Run tests

```
python3 tests.py
```

Read `logs/test_log.log` for test log.


