from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

# Intialization
classifier = Sequential()


# ------- CREATING THE CONVOLUTION LAYER -------
# Using rectifier activiation function to ensure non-lineararity and to remove negative pixels
classifier.add(Convolution2D(32, 3, 3, input_shape = (64,64,3), activation = "relu"))

# ------ MAX POOLING ------
# 2 by 2 ensures we still maintain information of the features
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# -------- ADDING SECOND CONVOLUTIONAL LAYER ------
classifier.add(Convolution2D(32, 3, 3, activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))



# ------- FLATTENING -------
classifier.add(Flatten())


# ------- FULL CONNECTION TO ANN -------
classifier.add(Dense(output_dim = 128, activation = "relu"))
# Using sigmoid function for the output since we have a binary outcome (either cat or dog)
# If not binary, use softmax activation function
classifier.add(Dense(output_dim = 1, activation = "sigmoid"))


# ----- COMPILING ------
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# -------- IMAGE PREPROCESSING -------
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)



# ----------- CODE FOR SINGLE PREDICTIONS --------
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
# Making the image into a 3D array
test_image = image.img_to_array(test_image)
# Adding a 4th dimension to repersent the batch of inputs (for this case, batch of size 1), as the predict method wants
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'