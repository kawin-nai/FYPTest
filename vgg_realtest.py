import keras_vggface
import mtcnn
import tensorflow as tp
import numpy as np
import os
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import mtcnn
from PIL import Image
from scipy.spatial import distance


def extract_face(img_path, detector=mtcnn.MTCNN(), required_size=(224, 224)):
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
        face_image = face_image.resize((224, 224))
        face_array = np.asarray(face_image)
        face_images.append(face_array)
    return face_images


def draw_faces(path, faces):
    # draw each face separately
    for i in range(len(faces)):
        plt.subplot(1, len(faces), i + 1)
        plt.axis('off')
        plt.imshow(faces[i])
    plt.show()


img_count = 0
face_count = 0
img_path = ".\\content\\CMFD"
for (root, dir, files) in os.walk(img_path):
    for file in files:
        path = os.path.join(root, file)
        faces = extract_face(path)
        # if faces is not None:
        #     draw_faces(path, faces)

# Print the number of images and faces
print("Total images: ", img_count)
print("Total faces: ", face_count)
