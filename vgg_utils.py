import keras_vggface
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mtcnn
from scipy.spatial.distance import cosine
from sklearn.model_selection import train_test_split
from PIL import Image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from scipy.spatial import distance
import cv2


def extract_faces(img_path, detector, required_size=(224, 224)):
    global face_count
    global img_count
    img = plt.imread(img_path)
    img_count += 1
    faces = detector.detect_faces(img)
    print(img_path, len(faces))
    if not faces:
        return None
    face_count += 1
    # Extract the list of faces in a photograph
    face_images = []
    for face in faces:
        print(face)
        # extract the bounding box from the requested face
        x1, y1, width, height = face['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        # extract the face
        face_boundary = img[y1:y2, x1:x2]
        plt.imshow(face_boundary)
        # resize pixels to the model size
        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize(required_size)
        face_array = np.asarray(face_image)
        face_images.append(face_array)
    return face_images


def extract_face(path, detector, required_size=(224, 224)):
    img = plt.imread(path)
    faces = detector.detect_faces(img)
    if not faces:
        return None
    # extract details of the largest face
    x1, y1, width, height = faces[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face_boundary = img[y1:y2, x1:x2]
    # resize pixels to the model size
    face_image = Image.fromarray(face_boundary)
    face_image = face_image.resize(required_size)
    face_array = np.asarray(face_image)
    return face_array


def draw_faces(path, faces):
    # draw each face separately
    for i in range(len(faces)):
        plt.subplot(1, len(faces), i + 1)
        plt.axis('off')
        plt.imshow(faces[i])
    plt.show()


def get_embeddings(filenames, detector, model):
    # extract largest face in each filename
    faces = [extract_face(f, detector) for f in filenames]
    # convert into an array of samples
    try:
        samples = np.asarray(faces, 'float32')
        # prepare the face for the model, e.g. center pixels
        samples = preprocess_input(samples, version=2)
        # create a vggface model
        # model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        # perform prediction
        yhat = model.predict(samples)
        return yhat
    except:
        return None


def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
    else:
        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
    return score