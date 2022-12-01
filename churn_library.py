# library doc string


# import libraries
import logging
import os
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataframe_image as dfi
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    return pd.read_csv(pth)


def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """

    # Helper function to save images in specified path
    def save_img(pth):
        plt.savefig(pth, bbox_inches='tight')
        plt.clf()

    # Save dataframe description as image
    dfi.export(
        df.describe(),
        "./images/eda/dataset_description.png",
        table_conversion="matplotlib")
    # Plot a histogram of churned vs existing costumers
    df['Attrition_Flag'].hist()
    save_img("./images/eda/churn_dist")
    # Plot a histogram of customers age
    df['Customer_Age'].hist()
    save_img("./images/eda/customers_age")
    # Draw a normalized bar plot of customers martial status
    df.Marital_Status.value_counts('normalize').plot(kind="bar")
    save_img("./images/eda/martial_status_percentage")
    # distplot is deprecated. Use histplot instead
    # sns.distplot(df['Total_Trans_Ct']);
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained using a kernel density estimate
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    save_img("./images/eda/total_trans_ct")
    # Draw a correlotion heat map
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    save_img("./images/eda/features_correlations")


def encoder_helper(df, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    # Change the Attrition_Flag feature to Churn for a more meaningful name and encode it
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    for feature in category_lst:
        lst = []
        groups = df.groupby(feature).mean()[response]
        for val in df[feature]:
            lst.append(groups.loc[val])
        df["{}_Churn".format(feature)] = lst


def perform_feature_engineering(df, response):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    encoder_helper(df, category_lst, response)
    X = pd.DataFrame()
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                 'Income_Category_Churn', 'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]
    y = df[response]
    return train_test_split(X, y, test_size=0.3, random_state=42)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    pass


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importance in pth
    input:
            model: model object containing feature_importance_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig()


def train_models(X, X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    plot_model_results(y_test, y_test_preds_rf, y_train, y_train_preds_rf, 'Random Forest')
    plot_model_results(y_test, y_test_preds_lr, y_train, y_train_preds_lr, 'Logistic Regression')

    feature_importance_plot(cv_rfc.best_estimator_, X, "")


def plot_model_results(test_data, model_test_predictions, train_data, model_train_predictions, model_name):
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str(model_name + ' Train'), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(test_data, model_test_predictions)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str(model_name + ' Test'), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(train_data, model_train_predictions)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig()


if __name__ == "__main__":
    df = import_data("data/bank_data.csv")
    dfi.export(
        df.describe(),
        "./images/eda/dataset_description.png",
        table_conversion="matplotlib")
