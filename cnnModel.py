from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Dense, MaxPooling2D, Flatten

trainDataGen = ImageDataGenerator(rescale = 1./255, zoom_range = 0.2, shear_range = 0.2, horizontal_flip = True)
testDataGen = ImageDataGenerator(rescale = 1./255, zoom_range = 0.2, shear_range = 0.2, horizontal_flip = True)

trainX = trainDataGen.flow_from_directory('./Dataset/Garbage classification', target_size=(300,300), batch_size=32, class_mode='categorical')
testX = testDataGen.flow_from_directory('./Dataset/Garbage classification', target_size=(300,300), batch_size=32, class_mode='categorical')

print(trainX.class_indices)

#Building the model
model= Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(6,activation='softmax'))

model.summary()
model.compile(loss ='categorical_crossentropy',optimizer = "adam",metrics = ["accuracy"])

batch_size = 32
step_size_train=trainX.n//trainX.batch_size
step_size_test=testX.n//testX.batch_size
model.fit_generator(trainX, steps_per_epoch = step_size_train,epochs = 100,validation_data = testX,validation_steps = step_size_test)

model.save('garbageClassification.h5')
