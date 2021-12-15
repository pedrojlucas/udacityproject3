# Capstone project - Udacity Machine Learning Engineer using Azure nanodegree

This is the final assigment of the Udacity Machine Learning using Azure nanodegree course. It is intended to show all the skills and techniques learned during this course, as training a model, deploy it and finally consume it as an endpoint all of that using the Microsoft Azure Cloud framework.

In this project I will get a public domain dataset from Kaggle.com and I will train a model using two different methods provided in the Azure ML framework, the first one will be AutoML and the second one will be an standard Scikit-Learn model which hyperparameters will be tuned using Hyperdrive. I will create the necessary experiments for this training and hyperparameter tunings in both cases using the Azure ML Python SDK. Once I have the models trained I will deploy one of them as a webservice and use it to generate some predictions.

An scheme of the architecture of this project is shown:

![Architecture_scheme](/screenshots/Architecture_scheme.jpg)

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

![automl_auc_trend](/screenshots/automl_auc_trend.jpg)

The best model is the 'VotingEnsemble' model with an AUC of 0,936.

![automl_best_model](/screenshots/automl_bestmodel_runid.jpg)

In the previous screenshots we can see other metrics apart from the AUC weighted used for measure the perfomance of our model, we can see the accuracy and the precision too.

I have deployed this model as a webservice so it can be consumed for predictions, I am showing this deployment process in a separate chapter in this readme file.

## Hyperparameter Tuning

In this case I have chosen a Logistic Regression model for our classification task, I have tried to maximize the metric AUC weighted as I have done previosuly with the AutoML experiment. For achieving the best hyperparameters I have used Hyperdrive with the following parameters for its configuration:

![Hyperdrive_config](/screenshots/Hyperdrive_config.jpg)

I have configured a random sampling for getting the best hyperparameters of the model, in this case the hyperparameters optimized are the maximum number of iterations (max_iter) and the inverse of regularization strength (C). This random sampling is able to get good results within a reasonable time and with not consuming too many compute resources.

I have also set up a early stopping policy, in this case a bandit early stopping policy that will avoid taking too much time testing optimizations that will not get improvements over the best found solution until that moment.

### Results

I have run the Hyperdrive experiment in order to get the best hyperparameters for this Logistic Regression model:

![hyperdrive_runwidget](/screenshots/Hyperdrive_runwidget.jpg)

The model with the best hyperparameters gets an AUC weighted metric of 0,828. The best hyperparameters are shown in the next screenshot:

![hyperdrive_bestparameters](/screenshots/Hyperdrive_bestmodelparams.jpg)

And finally, as we have also done with our AutoML model, we have registered the model in the Azure ML environment:

![hyperdrive_modelregistered](/screenshots/Hyperdrive_bestmodelUI.jpg)

## Model Deployment

I have done two models deployment, one in a custom environment (this is the case of the AutoML model described previously) and the other one using a Scikit-Learn default environment available in Azure ML (in this case is for the Logistic Regression tuned with Hyperdrive).

The custom environment is provisioned with all the necessary dependencies and using a custom score python script to initialize the model webservice and to make the predictions. The conda dependecies setup file (in YAML format) is generated on the fly when the notebook is executed.

With the model registered, the environment allocated and the python score script prepared, I can proceed with the model deploy using the proper configuration options:

![automl_model_deploy_config](/screenshots/automl_model_deploy_config.jpg)

Once the deployment process finishes we can see the model endpoint as active:
![automl_active_endpoint](/screenshots/automl_endpoint_active.jpg)

An after I have generated the webservice for the model endpoint, I can test it with some data from our dataset. As we can see in the following screenshot we get a response from the webservice with two predictions.

![automl_webservice_response](/screenshots/automl_webservice_response.jpg)

I have also tested the deployment in a Azure default environment with Scikit-learn (more information about Azure curated environments here: https://docs.microsoft.com/en-us/azure/machine-learning/concept-prebuilt-docker-images-inference#list-of-prebuilt-docker-images-for-inference). In this case a score script is not needed and a YAML configuration file either, so all this work is done by Azure only specifying the model_framework option as Model.Framework.SCIKITLEARN on the model registering configuration.

So doing the model deployment in this way I have also got an active webservice endpoint for the Scikit-learn model tuned by Hyperdrive:

![hyperdrive_endpoint](/screenshots/Hyperdrive_endpoint_active.jpg)

And I have been able to check the model endpoint getting some predictions, in this case the returned predictions come in the form of probabilites of belonging to one of the two labels I am predicting with the model.

![hyperdrive_predictions](/screenshots/Hyperdrive_webservice_response.jpg)

## Screen Recording

I have recorded a video showing the different results obtained for the AutoML model. The link for the video on Youtube is https://youtu.be/LB6NQAejPRk

## Standout Suggestions

I have implemented some additional logs for the deployed models. I have activated the 'Application Insights' as an option for the deployment of the model, and in the scoring script, used for the initialization and serving of the model, I have put some print statements for showing in the logs different information as: model initialization success or failure, start of the inference process and end of inference process and also the results.

We can see here a couple of screenshots showing this logs:

![Webservice init logs](/screenshots/automl_logs_webservice1.jpg)

![Webservice_inference_logs](/screenshots/automl_logs_webservice2.jpg)

## Next steps and improvements

From the point of view of achieving better models, there are several improvements that can be tried for the AutoML model and for the Hyperdrive optimized model:

In order to get a better model:

- Try some feature engineering techniques.
- Rise the number of iterations.
- Get more data from other sources to complement the current dataset and increase the accuracy of the model.

In the case of the Scikit-learn model we can also try more sophisticated models based on decision trees for instance or maybe a Support Vector Machines to compare the results and check if we increase the primary metric.

In the other hand if we would like to enhance the architecture we can try:

- If we need a more scalable environment, maybe is necessary to deploy the models using an AKS (Azure Kubernetes System) instead of the used ACI.
- If we are going to put this models in production we will need a more user friendly interface to consume the models, like a web page that, once the data is provided, would call the model webservice endpoint.
- In order to get models that can be more deployable and architecture independent, we will need to export then as ONNX format.


