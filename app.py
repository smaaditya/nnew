import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load the model
model = load_model('weights.h5')

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((128, 128))  # Resize the image
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Function for simple chatbot responses
def chatbot_response(user_input):
    responses = {
        "hi": "Hello! How can I assist you today?",
        "what is this app?": "This app classifies images as malignant or not.",
        "help": "You can upload an image for classification.",
        "bye": "Goodbye! Have a great day!"
    }
    return responses.get(user_input.lower(), "I'm sorry, I don't understand that.")

# Streamlit app layout
st.title("Image Classification with TensorFlow and Chatbot")
st.write("Upload an image to classify it as malignant or not.")

# Chatbot Interface
st.subheader("Chatbot")
user_input = st.text_input("You:", "")

if user_input:
    response = chatbot_response(user_input)
    st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=None)

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    
    # Preprocess the image
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image)

    # Make predictions
    prediction = model.predict(processed_image)
    class_label = "Malignant" if prediction[0] > 0.5 else "Not Malignant"  # Threshold for binary classification

    # Display the result
    st.write(f"Prediction: {class_label}")
