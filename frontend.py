# streamlit_app.py

import streamlit as st
import requests

st.title("Multimodal Price Prediction")

# Input text description
text = st.text_area("Enter product description")

# Input image upload
uploaded_file = st.file_uploader("Upload product image", type=["png", "jpg", "jpeg"])

if st.button("Predict Price"):
    if not text:
        st.error("Please enter text description.")
    elif not uploaded_file:
        st.error("Please upload an image file.")
    else:
        # Prepare files and data for POST request
        files = {
            "image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        data = {
            "text": text
        }

        # Call API
        with st.spinner("Predicting..."):
            try:
                response = requests.post("http://localhost:8000/predict", data=data, files=files)
                response.raise_for_status()
                price = response.json().get("predicted_price")
                st.success(f"Predicted Price: {price:.2f}")
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")
