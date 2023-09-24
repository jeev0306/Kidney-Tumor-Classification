import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array

st.title("Kidney Tumor Classification")

st.header("Please upload an Image")

file = st.file_uploader('',type=['jpeg','jpg','png'])

model = load_model("C:/Users/abc/OneDrive/Documents/model/model.h5")

def classification(file, model):
    img_path = file
    img = load_img(img_path, target_size=(224, 224))  # Adjust the target size if needed
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to match the training data preprocessing

    # Make predictions using the provided model
    prediction = model.predict(img_array)

    # Determine the class label based on the prediction
    if prediction[0, 0] > 0.5:
          # print("prediction",prediction[0,0])
          class_label = 'Tumor'
          prediction = prediction[0,0]
    else:
          class_label = 'Normal'
          prediction = prediction[0, 0]
    return class_label, prediction


if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)
    detect_button = st.button("detect")
    if detect_button:
        class_name, prediction = classification(file, model)
        # print(class_name)
        st.write("## {}".format(class_name))
        st.write("### score: {}".format(prediction))

