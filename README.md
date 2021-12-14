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

### Task

The task that I am developing is a classification task where I will try to ask the question "Are this patient going to suffer heart disease?" based on the different features present in the dataset.

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

### Access

I have downloaded the dataset from Kaggle to my github account and I have added it to this project repository so it is not needed to have a Kaggle account and the dataset can be accessed then in two ways:

1.- The dataset file can be downloaded with the rest of the files from the github repository and then uploaded to Azure ML account folder where it can be accessed and register the Dataset in Azure.
2.- Directly accessing the file through http request to https://github.com/pedrojlucas/udacityproject3/blob/main/heart.csv and create a TabularDataset object in Azure ML, as I have configured this repository as public.

In the notebooks in the repository I have used the way number 1 in order to configure this project as stand-alone and reproducible as possible.

## Automated ML

For the AutoML experiments I am following the next steps:

1. Create a compute cluster for all the works regarding the training of the models.
2. Access the dataset as described in the previous point of this readme.
3. Configure the AutoML experiment using the automl_settings and automl_config presented in the next paragraphs.
4. Run the experiment.

I am using the following configuration for the AutoML experiment, the automl_settings are the general parameters for this experiment as the primary metric, number of cross validation, number of iterations allowed and the early stopping, on the other hand the automl_config is providing what dataset is using, and where is executing the training of the models.

![automl_configuration](/screenshots/automl_config_details.jpg)
I am using Area Under the Curve (AUC) as primary metric, this metric has advantages over other metrics like accuracy for our classification task, as it does not depend on the balance between the labels I need to predict. I have also chosen 5 folds for the cross validation in order to assure that I have representative results.
I have selected some options to get a good trade-off between good results and not consume too many compute resources, those are: enable early stopping, iterations, experiment_timeout.

### Results
After running the experiment for the AutoML, we have the best model:

![automl_run_widget](/screenshots/automl_runwidget.jpg)

The best model is the 'VotingEnsemble' model with an AUC of 0.93.

![automl_best_model](/screenshots/automl_bestmodel_runid.jpg)

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording

I have recorded a video showing the different results obtained for the AutoML model. The link for the video on Youtube is https://youtu.be/LB6NQAejPRk

## Standout Suggestions

I have implemented some additional logs for the deployed models. I have activated the 'Application Insights' as an option for the deployment of the model, and in the scoring script, used for the initialization and serving of the model, I have put some print statements for showing in the logs different information as: model initialization success or failure, start of the inference process and end of inference process and also the results.

We can see here a couple of screenshots showing this logs:

![Webservice init logs](/screenshots/automl_logs_webservice1.jpg)

![Webservice_inference_logs](/screenshots/automl_logs_webservice2.jpg)
