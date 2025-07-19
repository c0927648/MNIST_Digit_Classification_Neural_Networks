import streamlit as st
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf

# Page setup
st.set_page_config(page_title="Digit Classifier", layout="centered")
st.title("Digit Classifier (k-NN + PCA, NN, CNN)")

# Load all models
@st.cache_resource
def load_models():
    knn_pca = pickle.load(open("knn_model.pkl", "rb"))
    nn_model = tf.keras.models.load_model("mnist_nn_dropout.keras")
    cnn_model = tf.keras.models.load_model("cnn_model.keras")
    return knn_pca, nn_model, cnn_model

knn_model, nn_model, cnn_model = load_models()

# Preprocessing
def preprocess_for_knn(img):
    img = img.resize((28, 28)).convert("L")
    return np.array(img).astype("float32").flatten().reshape(1, -1) / 255.0

def preprocess_for_nn(img):
    img = img.resize((28, 28)).convert("L")
    return np.array(img).astype("float32").flatten().reshape(1, 784) / 255.0

def preprocess_for_cnn(img):
    img = img.resize((28, 28)).convert("L")
    return np.array(img).astype("float32").reshape(1, 28, 28, 1) / 255.0

# Upload image
uploaded_file = st.file_uploader("Upload a 28x28 grayscale digit image", type=["png", "jpg", "jpeg"])

# Select model
model_choice = st.selectbox("Choose a model to run:", ["k-NN + PCA", "Neural Network (Dropout)", "CNN"])

# Make prediction
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Input Image", width=150)

    if model_choice == "k-NN + PCA":
        x_input = preprocess_for_knn(image)
        prediction = knn_model.predict(x_input)[0]
        st.success(f"k-NN + PCA Prediction: **{prediction}**")

    elif model_choice == "Neural Network (Dropout)":
        x_input = preprocess_for_nn(image)
        pred_probs = nn_model.predict(x_input)
        prediction = np.argmax(pred_probs)
        st.success(f"Neural Network Prediction: **{prediction}**")
        if st.checkbox("Show probability scores"):
            st.bar_chart(pred_probs[0])

    elif model_choice == "CNN":
        x_input = preprocess_for_cnn(image)
        pred_probs = cnn_model.predict(x_input)
        prediction = np.argmax(pred_probs)
        st.success(f"CNN Prediction: **{prediction}**")
        if st.checkbox("Show probability scores"):
            st.bar_chart(pred_probs[0])
