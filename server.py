from flask import Flask,request
from tensorflow.keras.models import Model
from tensorflow.keras.models import Model,Sequential,load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img


model=load_model("ResNet50_rice_last.h5")


app=Flask(__name__)
@app.route('/model',methods=['POST'])
def serve_model():
    requested_data=request.get_json(force=True)
    img=requested_data["img"]
    img=np.array(img).reshape(-1,224,224,3)
    prediction=[model.predict(img).argmax()]
    if prediction==[0]:
        return ("Plant is suffering from Bacterial Leaf Blight. Use balanced amounts of plant nutrients, especially nitrogen. Ensure good drainage of fields in conventionally flooded crops and nurseries. Keep fields clean. Remove weed hosts and plow under rice stubble straw rice ratoons and volunteer seedlings which can serve as hosts of bacteria.Allow fallow fields to dry in order to suppress disease agents in the soil and plant residues")
        
    elif prediction==[1]:
        return ("Plant is suffering from Brown Spot.Application of edifenphos, chitosan, iprodione, or carbendazim in the field is also advisable.Spray Mancozeb (2.0g/lit) or Edifenphos (1ml/lit) - 2 to 3 times at 10 - 15 day intervals.Grisepfulvin, Nystatin, Aureofungin, and similar antibiotics have been found effective in preventing primary seedling infection.")
    
    elif prediction==[2]:
        return ("Plant is Healthy")
    
    elif prediction==[3]:
        return ("Plant is suffering from Hispa.Avoid over fertilizing the field.Close plant spacing results in greater leaf densities that can tolerate higher hispa numbers.Leaf tip containing blotch mines should be destroyed.Manual collection and killing of beetles – hand nets.To prevent egg laying of the pests, the shoot tips can be cut.Clipping and burying shoots in the mud can reduce grub populations by 75 - 92%.Spraying of methyl parathion 0.05% or Quinalphos 0.05%.")
    
    else:
        return ("Plant is suffering from LeafBlast.Prepare to use fungicide,Systemic fungicides like triazoles and strobilurins can be used to control blast.Silicon fertilizers (e.g., calcium silicate) can be applied to soils that are silicon deficient to reduce blast.Avoid excessive nitrogen application rates and apply no more than 30 pounds per acre of nitrogen per application at midseason.")
       
    


if  __name__ == "__main__":
    app.run()
