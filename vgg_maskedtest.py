import keras_vggface
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


def extract_faces(img_path, detector=mtcnn.MTCNN(), required_size=(224, 224)):
    # global face_count
    # global img_count
    img = plt.imread(img_path)
    # img_count += 1
    faces = detector.detect_faces(img)
    print(img_path, len(faces))
    if not faces:
        return None
    # face_count += 1
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


def extract_face(path, detector=mtcnn.MTCNN(), required_size=(224, 224)):
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


# Root image paths
root_path = "./content/mfr2/"

# Read pairs and split into matched_pairs and mismatched_pairs
pairs = pd.read_csv(root_path + "pairs.csv", sep=" ", header=None, names=["name1", "path1", "name2", "path2"])
matched_pairs = pairs[pairs["path2"].isnull()].drop("path2", axis=1)
mismatched_pairs = pairs[pairs["path2"].notnull()]

# Format paths
matched_pairs.rename(columns={"name2": "path2"}, inplace=True)
matched_pairs["path2"] = matched_pairs["path2"].astype(int)
mismatched_pairs["path2"] = mismatched_pairs["path2"].astype(int)

# Replace integer with image paths
matched_pairs["path1"] = root_path + matched_pairs["name1"] + "/" + matched_pairs["name1"] + "_" + matched_pairs[
    "path1"].apply(lambda x: '{0:0>4}'.format(x)) + ".png"
matched_pairs["path2"] = root_path + matched_pairs["name1"] + "/" + matched_pairs["name1"] + "_" + matched_pairs[
    "path2"].apply(lambda x: '{0:0>4}'.format(x)) + ".png"
mismatched_pairs["path1"] = root_path + mismatched_pairs["name1"] + "/" + mismatched_pairs["name1"] + "_" + \
                            mismatched_pairs["path1"].apply(lambda x: '{0:0>4}'.format(x)) + ".png"
mismatched_pairs["path2"] = root_path + mismatched_pairs["name2"] + "/" + mismatched_pairs["name2"] + "_" + \
                            mismatched_pairs["path2"].apply(lambda x: '{0:0>4}'.format(x)) + ".png"

# Split into train and test
matched_pairs_train, matched_pairs_test = train_test_split(matched_pairs, test_size=0.2, random_state=42)
mismatched_pairs_train, mismatched_pairs_test = train_test_split(mismatched_pairs, test_size=0.2, random_state=42)

# Create train and test sets
train = pd.concat([matched_pairs_train, mismatched_pairs_train])
test = pd.concat([matched_pairs_test, mismatched_pairs_test])

# Evaluate model
def matched_pair_evaluate(matched_pairs):
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    detector = mtcnn.MTCNN()
    scores = []
    same = []
    for i in range(len(matched_pairs)):
        filenames = [matched_pairs.iloc[i]["path1"], matched_pairs.iloc[i]["path2"]]
        print(filenames)
        embeddings = get_embeddings(filenames, detector, model)
        if embeddings is not None:
            score = is_match(embeddings[0], embeddings[1])
            scores.append(score)
            same.append((lambda a: a <= 0.5)(score))
        else:
            print("No face detected")
            scores.append(None)
            same.append(None)
    matched_pairs["score"] = scores
    matched_pairs["same"] = same
    # Calculate accuracy
    accuracy = len(matched_pairs[matched_pairs["same"] == True]) / len(matched_pairs)
    print("Accuracy: ", accuracy)
    # Save to csv
    matched_pairs.to_csv(root_path + "matched_pairs_eval.csv", index=False)


faces = extract_faces("./content/mfr2/AdrianDunbar/AdrianDunbar_0002.png", mtcnn.MTCNN())
draw_faces("./content/mfr2/AdrianDunbar/AdrianDunbar_0002.png", faces)
# matched_pair_evaluate(matched_pairs_test)
