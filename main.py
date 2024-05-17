import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models


# (training_images, training_images), (testing_images,
#                                      testing_labels) = datasets.cifar10.load_data()

# print("testinlabel : ", testing_images[:5])


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


dataset_path = "./datasets"
datasets = load_images_from_folder(dataset_path)
training_images = []
datasets_labels = []

for index, dataset in enumerate(datasets):
    print("dataset : ", index, dataset)
    for image in dataset:
        resized_image = cv.resize(image, (125, 125))

# training_images, testing_images = training_images/255, testing_images/255


model = models.Sequential()
# We add convolution layer with 32 filters filtering 3 per 3 pixels from images, input shape is 32x32x3 for basics
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# We add MaxPooling2D to filter the max between the array by going 2 by 2 from width to height, be careful
# at how TF invert array shape
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
