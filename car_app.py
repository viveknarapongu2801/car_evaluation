import streamlit as st
import pickle
import pandas as pd
import numpy as np
import category_encoders as ce

# ----------------------------
# Load the trained model
# ----------------------------
with open('car_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ----------------------------
# Prepare and fit the ordinal encoder
# ----------------------------
# Define all possible categories for each feature
categories = {
    'buying': ['vhigh', 'high', 'med', 'low'],
    'maint': ['vhigh', 'high', 'med', 'low'],
    'persons': ['2', '4', 'more'],
    'lug_boot': ['small', 'med', 'big'],
    'safety': ['low', 'med', 'high']
}

# Create a dummy DataFrame with one row per category (needed to fit encoder)
dummy_data = pd.DataFrame({
    'buying': categories['buying'],
    'maint': categories['maint'],
    'persons': categories['persons'] * 2,  # repeat to match length
    'lug_boot': categories['lug_boot'] * 3,
    'safety': categories['safety'] * 2
})

# Initialize the encoder and fit on dummy data
encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'persons', 'lug_boot', 'safety'])
encoder.fit(dummy_data)

# ----------------------------
# Streamlit App
# ----------------------------
st.title("Car Evaluation Prediction")
st.write("Enter the car features to get a prediction:")

# Input fields
buying = st.selectbox("Buying Price", categories['buying'])
maint = st.selectbox("Maint Price", categories['maint'])
door = st.selectbox("Number of Doors", ['2', '3', '4', '5more'])  # will be dropped
persons = st.selectbox("Number of Persons", categories['persons'])
lug_boot = st.selectbox("Luggage Boot Size", categories['lug_boot'])
safety = st.selectbox("Safety", categories['safety'])

# Prediction
if st.button("Predict"):
    # Create DataFrame from user input
    input_data = pd.DataFrame([[buying, maint, door, persons, lug_boot, safety]],
                              columns=['buying', 'maint', 'door', 'persons', 'lug_boot', 'safety'])
    
    # Drop 'door' if it was not used in training
    input_data = input_data.drop('door', axis=1)
    
    # Encode
    input_encoded = encoder.transform(input_data)
    
    # Predict
    prediction = model.predict(input_encoded)
    
    st.success(f"Predicted Car Evaluation: {prediction[0]}")
