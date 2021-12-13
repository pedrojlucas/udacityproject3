import json
import pandas as pd
import joblib
from azureml.core.model import Model

cols = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

def init():
    global model

    # depending on whether we use automl/hyperdrive, uncomment accordingly
    try: 
        # model_path = Model.get_model_path('my-udacityproj3-hyperdrivemodel')
        model_path = Model.get_model_path('my-udacityproj3-automlmodel')
        model = joblib.load(model_path)
    except Exception as err:
        print('init method error: ' + str(err))

def run(data):
    try:
        # model_path = Model.get_model_path('my-udacityproj3-hyperdrivemodel')
        model_path = Model.get_model_path('my-udacityproj3-automlmodel')
        model = joblib.load(model_path)
        data = pd.DataFrame(json.loads(data)['data'], columns=cols)
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = 'Run Error: ' + str(e)
        return error