import keras
import keras_vggface
import mtcnn
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import mtcnn
from scipy.spatial.distance import cosine
from sklearn.model_selection import train_test_split
from PIL import Image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from scipy.spatial import distance
from tensorflow.python.keras import layers


# Create a facial recognition model from scratch
def create_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(model, train_data, train_labels, test_data, test_labels):
    model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))


model = create_model()
model.summary()
train_model(model, train_data, train_labels, test_data, test_labels)
