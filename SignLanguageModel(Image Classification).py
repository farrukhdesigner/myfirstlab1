# import all required libraries

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# path to dataset
path = r"C:\Users\farru\Downloads\Compressed\ai sign language\sign-text dataset-20230830T205109Z-001\sign-text dataset\ImagePro"
# list of files in path
files = os.listdir(path)


# sort
files.sort()

# print to see list
print(files)


# list of image
image_array=[]
# list of labels
label_array=[]

# looping through each file in files

for i in tqdm(range(len(files))):
	# list of image in each folder
	sub_folder_path = os.path.join(path, files[i])
	sub_file = os.listdir(sub_folder_path)
	# check length of each folder
	#print(len(sub_file))

	# loop through each sub folder
	for j in range(len(sub_file)):

		# path of each image
		#Example:imagepro/A/image_name1.jpg

		file_path=path+"/"+files[i]+"/"+sub_file[j]
		# read each image

		image=cv2.imread(file_path)

		# resize image by 96x96
		image=cv2.resize(image,(96,96))
		# convert BGR image to RGB image
		image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

		# add this image at image_array
		image_array.append(image)

		# add label to label_array
		# i is number from 0 to len(files)-1
		# so we can use it as label
		label_array.append(i)

# convert list to array
image_array=np.array(image_array)
label_array=np.array(label_array,dtype="float")
print(len(label_array))

# split the dataset into test and train
from sklearn.model_selection import train_test_split

# X_train will have 85% of images 
# X_test will have 15% of images
X_train,X_test,Y_train,Y_test=train_test_split(image_array,label_array,test_size=0.15)

del image_array,label_array

# to free memory 
import gc
gc.collect()

# Create a model
from keras import layers,callbacks,utils,applications,optimizers
from keras.models import Sequential, Model, load_model

model=Sequential()

pretrained_model=tf.keras.applications.EfficientNetB0(input_shape=(96,96,3),include_top=False)
model.add(pretrained_model)

# add Pooling to model
model.add(layers.GlobalAveragePooling2D())

# add dropout to model
model.add(layers.Dropout(0.3))

# finally we will addd dense layer as an output
model.add(layers.Dense(1))

model.build(input_shape=(None,96,96,3))


# to see model summary
model.summary()

# compile model
model.compile(optimizer="adam",loss="mae",metrics=["mae"])

# create a checkpoint to save best accuracy model
ckp_path="trained_model"
model_checkpoint=tf.keras.callbacks.ModelCheckpoint(
							filepath=ckp_path,
							monitor="val_mae",
							mode="auto",
							verbose=1,
							save_best_only=True,
							save_weights_only=True
							)

# create learning rate reducer to reduce lr when accuracy does not improve
reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(
									factor=0.9,
									monitor="val_mae",
									mode="auto",
									cooldown=0,
									patience=5,
									verbose=1,
									min_lr=1e-6)



Epochs=1
Batch_Size=32
# Start training model
history=model.fit(
				X_train,
				Y_train,
				validation_data=(X_test,Y_test),
				batch_size=Batch_Size,
				epochs=Epochs,
				callbacks=[model_checkpoint,reduce_lr]
				)

# after the training is done load best model
model.load_weights("Models/Model Codes/Sign Language Model/Check points/trained_model/model")

model.load_weights(ckp_path)


# convert model to tensorflow lite model
converter=tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model=converter.convert()

# save model
with open("model.tflite","wb") as f:
	f.write(tflite_model)

# to see prediction result on test dataset
prediction_val=model.predict(X_test,batch_size=32)

# print first 10 values
print(prediction_val[:10])
# print first 10 values of Y_test
print(Y_test[:10])
