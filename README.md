# Capstone project - Udacity Machine Learning Engineer using Azure nanodegree

This is the final assigment of the Udacity Machine Learning using Azure nanodegree course. It is intended to show all the skills and techniques learned during this course, as training a model, deploy it and finally consume it as an endpoint all of that using the Microsoft Azure Cloud framework.

In this project I will get a public domain dataset from Kaggle.com and I will train a model using two different methods provided in the Azure ML framework, the first one will be AutoML and the second one will be an standard Scikit-Learn model which hyperparameters will be tuned using Hyperdrive. I will create the necessary experiments for this training and hyperparameter tunings in both cases using the Azure ML Python SDK. Once I have the models trained I will deploy one of them as a webservice and use it to generate some predictions.

## Project Set Up and Installation

As I stated before I will use the Azure ML Python SDK through Jupyter Notebooks. One notebook for the AutoML model training and deployment and other notebook for the scikit-learn model with additional hyperparameter tuning using Hyperdrive.

In the case of running the AutoML notebook is important to install the last version of all the Azure ML AutoML libraries to avoid conflicts during the analysis and serving of the AutoML model. This task is achieved executing the first cell of the Jupyter notebook and after that resetting the kernel of the Jupyter notebook to assure that the new libraries are loaded in the environment.

For the Hyperdrive tuning it is necessary to have an auxiliary python script for the model training and parsing the hyperparameters I am tuning. So the Hyperdrive experiment can change those hyperparameters and get the appropiate metrics to sort the best hyperparameters for the model selected.

## Dataset

### Overview

The used dataset for this project is a Kaggle public domain dataset that is focused in detecting the tendency to suffer a heart disease from certain features related with the health status of a number of patients. The url for accessing the dataset is https://www.kaggle.com/johnsmith88/heart-disease-dataset.

The label of the dataset is "Heartdisease" which can be 0 (no heart disease) or 1 (the patient suffers heart disease), and the different features that are included in the dataset:

1. age
2. sex
3. chest pain type (4 values)
4. resting blood pressure
5. serum cholestoral in mg/dl
6. fasting blood sugar > 120 mg/dl
7. resting electrocardiographic results (values 0,1,2)
8. maximum heart rate achieved
9. exercise induced angina
10. oldpeak = ST depression induced by exercise relative to rest
11. the slope of the peak exercise ST segment
12. number of major vessels (0-3) colored by flourosopy
13. thal: 0 = normal; 1 = fixed defect; 2 = reversable defect

### Task

The task that I am developing is a classification task where I will try to ask the question "Are this patient going to suffer heart disease?" based on the different features present in the dataset.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
