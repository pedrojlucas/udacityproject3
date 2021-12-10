import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://b46a4429-ae08-411f-ae50-9bf46c41a03c.southcentralus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = 'Fm437Ca4cmgKHX3l05FDi9r192s91n26'

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
            "Age": 55,
            "Sex": "M",
	        "ChestPainType": "ATA",	    
            "RestingBP": 170,	     
            "Cholesterol": 283,            
            "FastingBS": 0,
	        "RestingECG": "Normal",            
	        "MaxHR": 108,            
            "ExerciseAngina": "N",
            "OldPeak": 0,
            "ST_Slope": "Up",
          },
       ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())