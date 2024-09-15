import pickle
import numpy as np
import pandas as pd

loaded_model = pickle.load(open('/home/dhruv/Desktop/Machine Learning MOdels/Car_price_prediction/trained_model.sav', 'rb'))
input_data = {'make': 20, 'fule-type': 0, 'num-of-doors': 1, 'body-style': 0, 'num-of-cylinders': 4, 
'horsepower': 90, 'peak-rpm': 5500}

input_data_as_dict = pd.DataFrame([input_data])

reshaping_the_data = input_data_as_dict.to_numpy().reshape(-1,7)

prediction = loaded_model.predict(reshaping_the_data)*100