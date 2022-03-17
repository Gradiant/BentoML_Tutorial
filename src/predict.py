import requests
import numpy as np
import json
from loguru import logger

@logger.catch(reraise=True)
def predict():
    
    patient_data=[0.7, 1.0, 5.5, 0.6]

    logger.info("Sending patient data to BentoML service")

    response=requests.post(
        "http://127.0.0.1:3000/classify",
        headers={"content-type": "application/json"},
        data=str(patient_data)).text
    
    response=json.loads(response)
    
    for index, value in enumerate(response):
        logger.info(f"Patient data belongs to class {index} with {round(100*value, 2)}% probability")

if __name__ == "__main__":

    predict()