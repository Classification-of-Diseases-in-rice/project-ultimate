from flask import Flask,request
from tensorflow.keras.models import Model
from tensorflow.keras.models import Model,Sequential,load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img


model=load_model(r"Data_Set/Saved_Model/ResNet50_rice.h5")


app=Flask(__name__)
@app.route('/model',methods=['POST'])
def serve_model():
    requested_data=request.get_json(force=True)
    img=requested_data["img"]
    img=np.array(img).reshape(-1,224,224,3)
    return("the prediction is {}".format(['brownspot','healthy','bacterial leaf blight','hispa','tungro','leafblast'][model.predict(img).argmax()]))


if  __name__ == "__main__":
    app.run()