import os
import sys
import os
import dlib
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Print working directory
print(os.getcwd())
print(__file__)
# dir = os.path.dirname(__file__)
# print(dir)
dlib.DLIB_USE_CUDA = True
print(dlib.DLIB_USE_CUDA)
print(dlib.__version__)

shape_predictor_path = "./content/pretrained/shape_predictor_5_face_landmarks.dat"
model_path = "./content/pretrained/dlib_face_recognition_resnet_model_v1.dat"

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(shape_predictor_path)
model = dlib.face_recognition_model_v1(model_path)

# Specify image paths
img1_path = "./content/dataset/img1.jpg"
img2_path = "./content/dataset/img2.jpg"
img3_path = "./content/dataset/img3.png"

# Load the images in RGB
img1 = dlib.load_rgb_image(img1_path)
img2 = dlib.load_rgb_image(img2_path)
img3 = dlib.load_rgb_image(img3_path)

# Find frontal human faces in an image
# The second argument tells us that we upsample the image 1 time
img1_detected = detector(img1, 1)
img2_detected = detector(img2, 1)
img3_detected = detector(img3, 1)

# Get the landmarks of the faces (in this case, the first face detected)
img1_shape = sp(img1, img1_detected[0])
img2_shape = sp(img2, img2_detected[0])
img3_shape = sp(img3, img3_detected[0])

# Return the face as a Numpy array representing the image. The face will be rotated upright and scaled to 150x150 pixels
img1_aligned = dlib.get_face_chip(img1, img1_shape)
img2_aligned = dlib.get_face_chip(img2, img2_shape)
img3_aligned = dlib.get_face_chip(img3, img3_shape)

plt.subplot(1, 3, 1)
plt.imshow(img1_aligned)
plt.subplot(1, 3, 2)
plt.imshow(img2_aligned)
plt.subplot(1, 3, 3)
plt.imshow(img3_aligned)
plt.show()

img1_rep = model.compute_face_descriptor(img1_aligned)
img2_rep = model.compute_face_descriptor(img2_aligned)
img3_rep = model.compute_face_descriptor(img3_aligned)

distonetwo = distance.euclidean(img1_rep, img2_rep)
disttwothree = distance.euclidean(img2_rep, img3_rep)
distonethree = distance.euclidean(img1_rep, img3_rep)

print(distonetwo, disttwothree, distonethree)
