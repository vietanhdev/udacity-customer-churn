"""Customer Churn Predictor library
   Author: Viet-Anh Nguyen
   Created at: 24/01/2022
"""

from inspect import stack
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap
import errno
from os.path import join as path_join
import logging
import pathlib
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import os
import argparse
from constants import *

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()


class CustomerChurnPredictor:

    cat_columns = CAT_COLUMNS

    def __init__(self, data_path=None, log_path=None):
        """Initialize churn predictor
        input:
            data_path: path to data file
            log_path: path to log file
        """
        self.logger = logging.getLogger("CustomerChurnPredictor")
        if log_path is not None:
            log_path = pathlib.Path(log_path)
            log_folder = log_path.parent.absolute()
            if not os.path.isdir(log_folder):
                pathlib.Path(log_folder).mkdir(exist_ok=True, parents=True)

            file_logger = logging.FileHandler(log_path)
            NEW_FORMAT = '%(name)s - %(levelname)s - %(message)s'
            file_logger_format = logging.Formatter(NEW_FORMAT)
            file_logger.setFormatter(file_logger_format)
            self.logger.addHandler(file_logger)
            self.logger.setLevel(logging.INFO)

        # Data
        self.df = None
        self.X = None
        self.y = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

        # Models
        self.cv_rfc = None
        self.lrc = None

        if data_path is not None:
            self.import_data(data_path)

    def import_data(self, pth=DEFAULT_DATA_PATH):
        """Import data into customer churn predictor for training
        input:
            pth: a path to the csv
        """

        if not os.path.isfile(pth):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), pth)

        self.df = pd.read_csv(pth)

    def perform_eda(self, output_folder=DEFAULT_EDA_RESULT_PATH):
        """Perform EDA for data
        input:
            output_folder: output folder to save images
        """
        pathlib.Path(output_folder).mkdir(exist_ok=True, parents=True)

        if not os.path.isdir(output_folder):
            err = 'Output path must be a folder: {}'.format(output_path),
            exc_info = True
            self.logger.error(err)
            raise Exception(err)

        self.logger.info(
            "Performing EDA for data. Output folder: {}".format(output_folder))
        self.logger.info(self.df.head())
        self.logger.info("Data Shape: {}".format(self.df.shape))
        self.logger.info("NULL checking:")
        self.logger.info(self.df.isnull().sum())
        self.logger.info("Description:")
        self.logger.info(self.df.describe())

        self.logger.info("EDA for Churn")
        self.df['Churn'] = self.df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        plt.figure(figsize=(20, 10))
        self.df['Churn'].hist()
        churn_histogram_path = path_join(output_folder, "churn_histogram.png")
        plt.savefig(churn_histogram_path)
        plt.close('all')
        self.logger.info("Saved churn histogram into " + churn_histogram_path)

        self.logger.info("EDA for Customer Age")
        plt.figure(figsize=(20, 10))
        self.df['Customer_Age'].hist()
        customer_age_histogram_path = path_join(
            output_folder, "customer_age_histogram.png")
        plt.savefig(customer_age_histogram_path)
        plt.close('all')
        self.logger.info(
            "Saved customer age histogram into " +
            customer_age_histogram_path)

        self.logger.info("EDA for Marital Status")
        plt.figure(figsize=(20, 10))
        self.df.Marital_Status.value_counts('normalize').plot(kind='bar')
        marital_status_path = path_join(output_folder, "marital_status.png")
        plt.savefig(marital_status_path)
        plt.close('all')
        self.logger.info(
            "Saved marital status graph into " +
            marital_status_path)

        self.logger.info("EDA for Total_Trans_Ct")
        plt.figure(figsize=(20, 10))
        sns.distplot(self.df['Total_Trans_Ct'])
        total_trans_ct_path = path_join(output_folder, "total_trans_ct.png")
        plt.savefig(total_trans_ct_path)
        plt.close('all')
        self.logger.info(
            "Saved Total_Trans_Ct graph into " +
            total_trans_ct_path)

        self.logger.info("EDA for Correlations")
        plt.figure(figsize=(20, 10))
        sns.heatmap(self.df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        correlation_path = path_join(output_folder, "correlation.png")
        plt.savefig(correlation_path)
        plt.close('all')
        self.logger.info("Saved correlation graph into " + correlation_path)

    def encode_categorical_features(self, category_lst=cat_columns):
        """Turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook
        input:
            category_lst: list of columns that contain categorical features
        """
        for category in category_lst:
            lst = []
            groups = self.df.groupby(category).mean()['Churn']
            for val in self.df[category]:
                lst.append(groups.loc[val])
            self.df['{}_Churn'.format(category)] = lst

    def prepare_data(self, test_size=0.3):
        """Prepare and split data for training
        """

        if self.df is None:
            err = "Please import the data first!"
            self.logger.error(err)
            raise Exception(err)

        self.encode_categorical_features(self.cat_columns)

        self.y = self.df['Churn']
        self.X = pd.DataFrame()
        keep_cols = DATA_PREPARATION_KEEP_COLUMNS
        self.X[keep_cols] = self.df[keep_cols]

        # train test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42)

    def train_models(self, output_folder=DEFAULT_MODELS_PATH):
        """Train, store model results: images + scores, and store models
        input:
            output_folder: path to save models
        """

        if self.X_train is None or self.X_test is None \
                or self.y_train is None or self.y_test is None:
            err = "Please import and prepare the data first."
            self.logger.error(err)
            raise Exception(err)

        pathlib.Path(output_folder).mkdir(exist_ok=True, parents=True)

        # grid search
        self.rfc = RandomForestClassifier(random_state=42)
        self.lrc = LogisticRegression()

        param_grid = DEFAULT_PARAMS_GRID

        self.cv_rfc = GridSearchCV(
            estimator=self.rfc, param_grid=param_grid, cv=5)
        self.cv_rfc.fit(self.X_train, self.y_train)
        self.lrc.fit(self.X_train, self.y_train)

        # save best model
        joblib.dump(
            self.cv_rfc.best_estimator_,
            path_join(
                output_folder,
                'rfc_model.pkl'))
        joblib.dump(self.lrc, path_join(output_folder, 'logistic_model.pkl'))

    def evaluation(self, output_folder=DEFAULT_EVALUATION_PATH):
        """Produces classification report for training and testing results and stores report as image
        in images folder.
        input:
            output_folder: output folder to save images
        """
        if self.cv_rfc is None or self.lrc is None:
            err = "Please train a model."
            self.logger.error(err)
            raise Exception(err)

        pathlib.Path(output_folder).mkdir(exist_ok=True, parents=True)

        y_train_preds_rf = self.cv_rfc.best_estimator_.predict(self.X_train)
        y_test_preds_rf = self.cv_rfc.best_estimator_.predict(self.X_test)

        y_train_preds_lr = self.lrc.predict(self.X_train)
        y_test_preds_lr = self.lrc.predict(self.X_test)

        # scores
        self.logger.info('Random forest results')
        self.logger.info('Test results')
        self.logger.info(classification_report(self.y_test, y_test_preds_rf))
        self.logger.info('Train results')
        self.logger.info(classification_report(self.y_train, y_train_preds_rf))

        self.logger.info('Logistic regression results')
        self.logger.info('Test results')
        self.logger.info(classification_report(self.y_test, y_test_preds_lr))
        self.logger.info('Train results')
        self.logger.info(classification_report(self.y_train, y_train_preds_lr))

        lrc_plot = plot_roc_curve(self.lrc, self.X_test, self.y_test)
        save_path = path_join(output_folder, "lrc_roc_curve.png")
        plt.savefig(save_path)
        plt.close('all')
        self.logger.info('Saved lrc ROC curve to {}'.format(save_path))

        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        rfc_disp = plot_roc_curve(
            self.cv_rfc.best_estimator_,
            self.X_test,
            self.y_test,
            ax=ax,
            alpha=0.8)
        lrc_plot.plot(ax=ax, alpha=0.8)
        save_path = path_join(output_folder, "cv_rfc_roc_curve.png")
        plt.savefig(save_path)
        plt.close('all')
        self.logger.info('Saved cv_rfc ROC curve to {}'.format(save_path))

    def feature_importance_plot(self, save_path):
        """Creates and stores the feature importances in pth
        input:
            save_path: path to store the figure
        """
        # Calculate feature importances
        importances = self.cv_rfc.best_estimator_.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [self.X.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(self.X.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(self.X.shape[1]), names, rotation=90)
        plt.savefig(save_path)
        plt.close('all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training tool for Customer Churn.')
    parser.add_argument('--data', type=str, help='Input data path')
    parser.add_argument('--log_path', type=str, required=False, default="churn_library.log", help='Log path')
    parser.add_argument('--eda_output', type=str, required=True, help='EDA output path')
    parser.add_argument('--model_output', type=str, required=True, help='Model output path')
    parser.add_argument('--evaluation_output', type=str, required=False, help='Evaluation output path')
    parser.add_argument('--feature_importance_output', type=str, required=False, help='Feature importance output path')
    parser.add_argument('--test_size', type=float, required=False, default=0.3, help='Test ratio. Range: (0, 1.0)')

    args = parser.parse_args()

    predictor = CustomerChurnPredictor(args.data, log_path=args.log_path)
    predictor.perform_eda(output_folder=args.eda_output)
    predictor.prepare_data(test_size=args.test_size)
    predictor.train_models(output_folder=args.model_output)
    
    if args.evaluation_output is not None:
        predictor.evaluation(output_folder=args.evaluation_output)

    if args.feature_importance_output is not None:
        predictor.feature_importance_plot(args.feature_importance_output)