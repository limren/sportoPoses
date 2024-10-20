import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
# Possibilities
class_names = ['bench press', 'squat']

model_name = "sportsPosesClassifier.keras"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Optionally limit TensorFlow to only use a certain amount of GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected, running on CPU.")

if (not os.path.exists(model_name)):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        './datasets',
        target_size=(96, 96),
        batch_size=64,
        class_mode='sparse',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        './datasets',
        target_size=(96, 96),
        batch_size=64,
        class_mode='sparse',
        subset='validation'
    )

    print(train_generator.class_indices)

    # MODEL CONFIGURATION - 3 convolutions layers with MaxPooling & softmax function for predictions
    model = models.Sequential()
    # We add convolution layers with 32 filters filtering 3 per 3 pixels from images, input shape is 64x64x3 for basics
    # First Convolution Block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Second Convolution Block
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Third Convolution Block
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Fourth Convolution Block
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Global Average Pooling
    model.add(layers.GlobalAveragePooling2D())

    # Fully Connected Layers
    model.add(layers.Dropout(0.5))  # Dropout layer for regularization
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(len(class_names), activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_generator, epochs=12,
              validation_data=validation_generator)

    model.save("sportsPosesClassifier.keras")
else:
    model = models.load_model(model_name)


squatTest = cv.imread('squat.jpg')
squatTest = cv.cvtColor(squatTest, cv.COLOR_BGR2RGB)
squatTest = cv.resize(squatTest, (64, 64))


squatTest2 = cv.imread('squat2.jpg')
squatTest2 = cv.cvtColor(squatTest2, cv.COLOR_BGR2RGB)
squatTest2 = cv.resize(squatTest2, (64, 64))


benchPressTest = cv.imread('benchPress.jpg')
benchPressTest = cv.cvtColor(benchPressTest, cv.COLOR_BGR2RGB)
benchPressTest = cv.resize(benchPressTest, (64, 64))

predictionSquat = model.predict(np.array([squatTest])/255)
predictionSquat2 = model.predict(np.array([squatTest2])/255)
predictionBenchPress = model.predict(np.array([benchPressTest])/255)

indexSquat = np.argmax(predictionSquat)
indexSquat2 = np.argmax(predictionSquat2)
indexBenchPress = np.argmax(predictionBenchPress)

print("prediction squat :", class_names[indexSquat], indexSquat)
print("prediction squat2 :", class_names[indexSquat2], indexSquat2)
print("prediction bench press :",
      class_names[indexBenchPress], indexBenchPress)
