import pickle
import streamlit as st
import pandas as pd
import numpy as np

# Load the trained model
loaded_model = pickle.load(open('/home/dhruv/Desktop/Machine Learning MOdels/Car_price_prediction/trained_model.sav', 'rb'))

# Function to predict car price
def car_price_fun(input_data):
    # Convert categorical values to numerical values
    input_data['brand'] = options_brand[input_data['brand']]
    input_data['fuel'] = options_fuel[input_data['fuel']]
    input_data['doors'] = options_doors[input_data['doors']]
    input_data['body'] = options_body[input_data['body']]
    input_data['cylinders'] = options_cylinders[input_data['cylinders']]
    
    input_data_as_dict = pd.DataFrame([input_data])

    reshaping_the_data = input_data_as_dict.to_numpy().reshape(-1, 7)
    
    prediction = loaded_model.predict(reshaping_the_data) * 100
    return prediction

# Main function to create the Streamlit app
def main():
    st.title("Car Price Prediction Web App")
    st.header("Enter the Car Details")

    # Create a dictionary to store the input data
    input_data = {}

    global options_brand, options_fuel, options_doors, options_body, options_cylinders

    st.subheader("Brand Name")
    options_brand = {"Audi": 0, "BMW": 1, "Chevrolet": 2, "Dodge": 3, "HONDA": 4, "Jaguar": 5, "Mercedes-Benz": 6, "Nissan": 7, "Porsche": 8, "Tesla": 9, "Toyota": 10, "Volkswagen": 11, "Volvo": 12}
    input_data['brand'] = st.selectbox("Choose an option:", list(options_brand.keys()))

    st.subheader("Fuel Type")
    options_fuel = {"GAS": 0, "Diesel": 1}
    input_data['fuel'] = st.selectbox("Choose an option:", list(options_fuel.keys()))

    st.subheader("Number of Doors")
    options_doors = {"Four": 4, "Two": 2}
    input_data['doors'] = st.selectbox("Choose an option:", list(options_doors.keys()))

    st.subheader("Body Style")
    options_body = {"convertible": 0, "hatchback": 1, "sedan": 2, "wagon": 4, "hardtop": 3}
    input_data['body'] = st.selectbox("Choose an option:", list(options_body.keys()))

    st.subheader("Number of Cylinders")
    options_cylinders = {"four": 4, "six": 6, "five": 5, "eight": 8, "two": 2, "twelve": 12, "three": 3}
    input_data['cylinders'] = st.selectbox("Choose an option:", list(options_cylinders.keys()))

    st.subheader("HorsePower")
    input_data['horsepower'] = st.number_input("Enter HorsePower:", min_value=0, max_value=100000, value=50, key='horsepower')

    st.subheader("Peak RPM")
    input_data['rpm'] = st.number_input("Enter Peak RPM:", min_value=0, max_value=100000, value=50, key='rpm')

    # Prediction
    if st.button("Predict"):
        prediction = car_price_fun(input_data)
        st.success(f"The estimated price of the car is: ${prediction[0]:.2f}")

    # Reset
    if st.button("Reset"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()
