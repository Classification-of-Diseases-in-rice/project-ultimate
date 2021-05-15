import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
import json
import requests
import PIL
import matplotlib.pyplot as plt

img=np.array(load_img("_______").resize((224,224))).tolist()   #Testing image url in ______

url='http://127.0.0.1:5000/model'


requested_data=json.dumps({'img':img})
response = requests.post(url,requested_data)
response.text
 
  
