
# Installing necessary packages
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

# Pixel list is the list of actual 3D pixels on which the class predictions have to be mapped to
pixel_list = [(0,0,0), (255,255,0), (255,0,255), (0,255,0), (0,0,255), (255,255,255), (0,255,255)]

# To Avoid Warning
st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("DeepGlobe Landcover classification")

st.text("Upload an image")

# Storing the Models in Cache memory allows to avoid reloading the model agin and again
@st.cache(allow_output_mutation=True)
def load_models():
  model1 = load_model('/content/drive/MyDrive/vgg16_backbone.hdf5', compile=False)
  model2 = load_model('/content/drive/MyDrive/res34_backbone.hdf5', compile=False)
  model3 = load_model('/content/drive/MyDrive/incep_backbone.hdf5', compile=False)
  return model1, model2, model3

model1, model2, model3 = load_models()

# Defining the Prediction Function for the main segmentation
def predict_from_models(image1, model1, model2, model3):
  size = (512, 512)

  # (1)... Image Preprocessing
  image = ImageOps.fit(image1, size, Image.ANTIALIAS)
  image = np.asarray(image)
  image = image/255.0
  image = image[np.newaxis, ...]
  
  # (2) Model prediction
  predict1 = model1.predict(image)
  predict2 = model2.predict(image)
  predict3 = model3.predict(image)

  # (3) Image Reprocessing
  pred = np.squeeze((0.3 * predict1) + (0.2 * predict2) + (0.2 * predict3), axis=0)
  weighted_ensemble = np.argmax(pred, axis=-1)

  return weighted_ensemble

# Creating an UploadFile object to make predictions
file_up = st.file_uploader("Please Upload a Satellite Image", type=['jpg', 'png'])


if file_up is None:
  st.text("Please upload an image !!") 
else:
  image = Image.open(file_up)
  st.image(image, caption='Uploaded Image')
  st.write("Predicting...")
  output_img = predict_from_models(image, model1, model2, model3)
  
  real_img = np.zeros((512, 512, 3))

  # The Loop below is used to map the class predictions, to their respective pixel predictions
  for i in range(7):
    a, b = np.where(output_img == i)
    pxl = pixel_list[i]
  
    for j in range(len(a)):
      real_img[a[j], b[j]] = np.asarray(pxl)

  # Putting the image as output
  st.image(real_img, clamp=True, channels='RGB', caption = 'Predicted Output')
