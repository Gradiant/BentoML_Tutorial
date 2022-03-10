# service.py
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

model_name="tutorial_svm"
model_version="latest"
service_name="bentoml_tutorial"

model = bentoml.sklearn.load_runner(f"{model_name}:{model_version}")

clf = bentoml.Service(service_name, runners=[model])

@clf.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series) -> np.ndarray:

    input_series = np.array(input_series)    
    result = model.run(input_series)
    
    return result