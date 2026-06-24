import requests
import json
import time
import random

url = 'http://localhost:8000/predict'

headers = {'Content-Type': 'application/json'}

print(f"Mulai mengirim traffic ke {url}...")

while True:
    age = random.randint(20, 60)
    salary = random.randint(20000, 100000)
    
    payload = {
        "dataframe_split": {
            "columns": ["Age", "EstimatedSalary"],
            "data": [[age, salary]]
        }
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        print(f"Input: [{age}, {salary}] -> Response: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(1)