import streamlit as st
import pickle
import pandas as pd
import numpy as np
import category_encoders as ce

# Load the trained model
with open('car_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the ordinal encoder and fit it with the original training data
# This is important to ensure consistent encoding between training and deployment
# Assuming you have the original X_train data available or can recreate it
# For simplicity, I'll recreate a dummy encoder based on the columns and expected values
# In a real scenario, you would save and load the fitted encoder
# along with the model, or refit it on a representative dataset.

# Based on the value_counts() output, the order for encoding seems to be:
# buying, maint: vhigh, high, med, low
# door: 2, 3, 4, 5more
# persons: 2, 4, more
# lug_boot: small, med, big
# safety: low, med, high

# Define the mapping for each column based on the order found during exploration
mapping = [{'col': 'buying', 'mapping': {'vhigh': 1, 'high': 2, 'med': 3, 'low': 4, None: 0}},
           {'col': 'maint', 'mapping': {'vhigh': 1, 'high': 2, 'med': 3, 'low': 4, None: 0}},
           {'col': 'door', 'mapping': {'2': 1, '3': 2, '4': 3, '5more': 4, None: 0}},
           {'col': 'persons', 'mapping': {'2': 1, '4': 2, 'more': 3, None: 0}},
           {'col': 'lug_boot', 'mapping': {'small': 1, 'med': 2, 'big': 3, None: 0}},
           {'col': 'safety', 'mapping': {'low': 1, 'med': 2, 'high': 3, None: 0}}]


encoder = ce.OrdinalEncoder(mapping=mapping)

# Streamlit App
st.title("Car Evaluation Prediction")

st.write("Enter the car features to get a prediction:")

# Create input fields for each feature
buying = st.selectbox("Buying Price", ['vhigh', 'high', 'med', 'low'])
maint = st.selectbox("Maint Price", ['vhigh', 'high', 'med', 'low'])
door = st.selectbox("Number of Doors", ['2', '3', '4', '5more'])
persons = st.selectbox("Number of Persons", ['2', '4', 'more'])
lug_boot = st.selectbox("Luggage Boot Size", ['small', 'med', 'big'])
safety = st.selectbox("Safety", ['low', 'med', 'high'])

# Create a button to predict
if st.button("Predict"):
    # Create a pandas DataFrame from the input
    input_data = pd.DataFrame([[buying, maint, door, persons, lug_boot, safety]],
                              columns=['buying', 'maint', 'door', 'persons', 'lug_boot', 'safety'])

    # Encode the input data
    # Need to handle the case where 'door' was dropped during training but is in the app input
    # Based on the previous code, 'door' was dropped from the features used for training.
    # So, we need to drop 'door' from the input_data before encoding and prediction.
    input_data = input_data.drop('door', axis=1)

    input_encoded = encoder.transform(input_data)

    # Make prediction
    prediction = model.predict(input_encoded)

    # Display the prediction
    st.write(f"Predicted Car Evaluation: {prediction[0]}")
