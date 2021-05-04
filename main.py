from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization ,GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np


"""
changing image size & Preprocessing
"""

img_height,img_width = (224,224)
batch_size=32
train_data_dir=r"E:\PROJECT\Scripts\Data_Set\Processed_Data\train"
test_data_dir=r"E:\PROJECT\Scripts\Data_Set\Processed_Data\test"
valid_data_dir=r"E:\PROJECT\Scripts\Data_Set\Processed_Data\val"


train_datagen= ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.4)

train_generator=train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')#set as training data

valid_generator=train_datagen.flow_from_directory(
    valid_data_dir,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')#set as validation data

test_generator = train_datagen.flow_from_directory(
    test_data_dir,#same directory as training data
    target_size=(img_height,img_width),
    batch_size=1,
    class_mode='categorical',
    subset='validation')#set as validation data
 
x,y = test_generator.next()
x.shape

base_model= ResNet50(include_top=False,weights='imagenet')
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
predictions=Dense(train_generator.num_classes,activation='softmax')(x)
model= Model(inputs=base_model.input,outputs=predictions)


for layer in base_model.layers:
    layer.trainable=True


model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_generator,epochs=10)

model.save('Saved_Model\ResNet50_rice_last.h5')

test_loss,test_acc=model.evaluate(test_generator,verbose=2)

val_loss,val_acc=model.evaluate(valid_generator,verbose=2)

train_loss,train_acc=model.evaluate(train_generator,verbose=2)

print('\n Test Accuracy',test_acc)

print('\n Train Accuracy',train_acc)

print('\n Validation Accuracy',val_acc)
