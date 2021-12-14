import json
import pandas as pd
import joblib
from azureml.core.model import Model

cols = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

def init():
    global model

    # Capture the errors just in case to make easier the debugging.
    try: 
        print('Initializing model instance as webservice...')
        model_path = Model.get_model_path('my-udacityproj3-automlmodel')
        model = joblib.load(model_path)
        print('Init done... enjoy inferencing!!')
    except Exception as err:
        print('init method error: ' + str(err))

def run(data):
    try:
        print('Launching inference...')
        model_path = Model.get_model_path('my-udacityproj3-automlmodel')
        model = joblib.load(model_path)
        data = pd.DataFrame(json.loads(data)['data'], columns=cols)
        result = model.predict(data)
        print('Inference done...')
        print('Results from inference:', result.tolist())
        return result.tolist()
    except Exception as e:
        error = 'Run Error: ' + str(e)
        print(error)
        return error