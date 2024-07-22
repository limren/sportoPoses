import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models


# Possibilities
class_names = ['bench press', 'squat']

model_name = "sportsPosesClassifier.keras"



# model = models.load_model(model_name)

def load_images_from_folder(dataset_path):
    """
    Load images from a dataset folder containing multiple subfolders with images.

    :param dataset_path: Path to the dataset folder.
    :return: A dictionary where keys are subfolder names and values are lists of images.
    """
    data = {}

    for subdir in os.listdir(dataset_path):
        print("subdir : ", subdir)
        subdir_path = os.path.join(dataset_path, subdir)

        if os.path.isdir(subdir_path):
            images = []

            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)

                image = cv.imread(file_path)

                if image is not None:
                    images.append(image)

            # Store the images in the dictionary with the subdirectory name as key
            data[subdir] = images

    return data

if(not os.path.exists(model_name)):
    dataset_path = "./datasets"
    datasets = load_images_from_folder(dataset_path)
    validation_images = []
    validation_labels = []

    # We get the indexes of enumerate(datasets) : squat = 0 & benchPress = 1
    training_images = []
    training_labels = []

    for index, (dataset_name, images) in enumerate(datasets.items()):
        # print("Dataset:", index, dataset_name)
        dataset_path_resized = dataset_name + "_resized"
        fullPath = os.path.join(dataset_path, dataset_path_resized)
        print("index : ", index, dataset_name)
        # Create directory for resized images if it doesn't exist
        if not os.path.exists(fullPath):
            os.mkdir(fullPath)

        half = len(images)/4

        for i, image in enumerate(images):
            # Resize the image
            resized_image = cv.resize(image, (125, 125))

            if i>half:
                # Append to training data
                training_images.append(resized_image)
                training_labels.append([index])
            else:
                validation_images.append(resized_image)
                validation_labels.append([index])

            # Save the resized image
            image_filename = f"{dataset_name}_{i}.jpg"
            cv.imwrite(os.path.join(fullPath, image_filename), resized_image)

    # Convert training data to np arrays
    training_images = np.array(training_images)/255
    training_labels = np.array(training_labels)

    validation_images = np.array(validation_images)/255
    validation_labels = np.array(validation_labels)


    # MODEL CONFIGURATION - 3 convolutions layers with MaxPooling & softmax function for predictions
    model = models.Sequential()
    # We add convolution layer with 32 filters filtering 3 per 3 pixels from images, input shape is 32x32x3 for basics
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(125, 125, 3)))
    # We add MaxPooling2D to filter the max between the array by going 2 by 2 from width to height, be careful
    # at how TF invert array shape
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # I let the same for training & validation data for now
    model.fit(training_images, training_labels, epochs=7,
            validation_data=(validation_images, validation_labels))


    model.save("sportsPosesClassifier.keras")
else:
    model = models.load_model(model_name)




squatTest = cv.imread('squat.jpg')
squatTest = cv.cvtColor(squatTest, cv.COLOR_BGR2RGB)
squatTest = cv.resize(squatTest, (125, 125))


squatTest2 = cv.imread('squat2.jpg')
squatTest2 = cv.cvtColor(squatTest2, cv.COLOR_BGR2RGB)
squatTest2 = cv.resize(squatTest2, (125, 125))



benchPressTest = cv.imread('benchPress.jpg')
benchPressTest = cv.cvtColor(benchPressTest, cv.COLOR_BGR2RGB)
benchPressTest = cv.resize(benchPressTest, (125, 125))

predictionSquat = model.predict(np.array([squatTest])/255)
predictionSquat2 = model.predict(np.array([squatTest2])/255)
predictionBenchPress = model.predict(np.array([benchPressTest])/255)

indexSquat = np.argmax(predictionSquat)
indexSquat2 = np.argmax(predictionSquat2)
indexBenchPress = np.argmax(predictionBenchPress)

print("prediction squat :", class_names[indexSquat], indexSquat)
print("prediction squat2 :", class_names[indexSquat2], indexSquat2)
print("prediction bench press :", class_names[indexBenchPress], indexBenchPress)
