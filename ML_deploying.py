import numpy as np
import pickle


loaded_model = pickle.load(open('/home/dhruv/Desktop/Machine Learning MOdels/MIne Vs Rock Detection Model/trained_model.sav', 'rb'))

input_data = (0.0762,0.0666,0.0481,0.0394,0.0590,0.0649,0.1209,0.2467,0.3564,0.4459,0.4152,0.3952,0.4256,0.4135,0.4528,0.5326,0.7306,0.6193,0.2032,0.4636,0.4148,0.4292,0.5730,0.5399,0.3161,0.2285,0.6995,1.0000,0.7262,0.4724,0.5103,0.5459,0.2881,0.0981,0.1951,0.4181,0.4604,0.3217,0.2828,0.2430,0.1979,0.2444,0.1847,0.0841,0.0692,0.0528,0.0357,0.0085,0.0230,0.0046,0.0156,0.0031,0.0054,0.0105,0.0110,0.0015,0.0072,0.0048,0.0107,0.0094)
input_data_as_numpy = np.asarray(input_data)
reshaping_the_data = input_data_as_numpy.reshape(1,-1)
prediction = loaded_model.predict(reshaping_the_data)
if(prediction=='R'):
    print("[R]: It is a Rock : You Are Safe")
else:
    print("[M]: It is a Mine : Danger")
