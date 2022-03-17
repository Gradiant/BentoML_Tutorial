import requests
import numpy as np
from loguru import logger

@logger.catch(reraise=True)
def predict():
    
    patient_data=[0.7, 1.0, 5.5, 0.6]

    logger.info("Sending patient data to BentoML service")

    response=requests.post(
        "http://127.0.0.1:3000/classify",
        headers={"content-type": "application/json"},
        data=str(patient_data)).text
    
    response=np.fromstring(response.replace("[","").replace("]",""), dtype=float, sep=',')

    response = {k:v for k, v in enumerate(response)}
    for k, v in response.items():
        logger.info(f"Patient data belongs to class {k} with {round(100*v, 2)}% probability")

if __name__ == "__main__":

    predict()