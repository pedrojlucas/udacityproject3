from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

def clean_data(data):
    
    # Clean and one hot encode data
    x_df = data
    x_df['Age'] = (x_df['Age'] - np.min(x_df['Age'])) /(np.max(x_df['Age'])-np.min(x_df['Age']))
    x_df['RestingBP'] = (x_df['RestingBP'] - np.min(x_df['RestingBP']))/(np.max(x_df['RestingBP'])-np.min(x_df['RestingBP']))
    x_df['Cholesterol'] = (x_df['Cholesterol'] - np.min(x_df['Cholesterol']))/(np.max(x_df['Cholesterol'])-np.min(x_df['Cholesterol']))
    x_df['MaxHR'] = (x_df['MaxHR'] - np.min(x_df['MaxHR']))/(np.max(x_df['MaxHR'])-np.min(x_df['MaxHR']))
    x_df['Oldpeak'] = (x_df['Oldpeak'] - np.min(x_df['Oldpeak']))/(np.max(x_df['Oldpeak'])-np.min(x_df['Oldpeak']))
    chestpain = pd.get_dummies(x_df['ChestPainType'], prefix="ChestPain")
    x_df.drop("ChestPainType", inplace=True, axis=1)
    x_df = x_df.join(chestpain)
    x_df["Sex"] = x_df['Sex'].apply(lambda s: 1 if s == "M" else 0)
    x_df["ExerciseAngina"] = x_df['ExerciseAngina'].apply(lambda s: 1 if s == "Y" else 0)
    restingecg = pd.get_dummies(x_df['RestingECG'], prefix="RestingECG")
    x_df.drop("RestingECG", inplace=True, axis=1)
    x_df = x_df.join(restingecg)
    stslope = pd.get_dummies(x_df['ST_Slope'], prefix='ST_Slope')
    x_df.drop("ST_Slope", inplace=True, axis=1)
    x_df.join(stslope)

    y_df = x_df.pop("HeartDisease")

    return x_df, y_df
    

def main():
    
    # Create pandas dataframe from 'heart.csv' data file.

    ds = pd.read_csv('./heart.csv')

    x, y = clean_data(ds)

    # Split data into train and test sets.

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    run = Run.get_context()

    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    auc = roc_auc_score(y_pred, y_test, average='weighted')
    run.log("AUC", np.float(auc))

if __name__ == '__main__':
     
    main()
