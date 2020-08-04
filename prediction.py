from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("./garbageClassification.h5")
img = image.load_img('./uploads/cardboardPic.jpeg',target_size = (300,300))
x = image.img_to_array(img)
print(x.shape)

x = np.expand_dims(x,axis =0)
print(x.shape)

predItem = model.predict_classes(x)
print(predItem)

index = ['Cardboard','Glass','Metal','Paper','Plastic','Trash']
print("The item is : ",index[predItem[0]])
