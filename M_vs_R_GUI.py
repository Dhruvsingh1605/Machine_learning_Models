import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('/home/dhruv/Desktop/Machine Learning MOdels/MIne Vs Rock Detection Model/trained_model.sav', 'rb'))

def mine_vs_rock_fun(input_data):
    
    input_data_as_numpy = np.asarray(input_data)
    reshaping_the_data = input_data_as_numpy.reshape(1,-1)
    prediction = loaded_model.predict(reshaping_the_data)
    if(prediction=='R'):
        print("[R]: It is a Rock : You Are Safe")
    else:
        print("[M]: It is a Mine : Danger")

def main():
    st.title("Mine Vs Rock Detection Web App")

    # Getting the input for prediction

    st.title("Sonar values V!,V2,V3......")

    for i in range(60):
        input_data = st.text_input("V"+str(i+1))
    
    R_or_M = ""

    if st.button("Predict"):
        R_or_M = mine_vs_rock_fun([input_data])
    
    st.success(R_or_M)


if __name__ == "__main__":
    main()