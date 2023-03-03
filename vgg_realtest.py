import json

import vgg_scratch
from vgg_utils_withsave import *
from vgg_scratch import *

# A file that tests the model on every available pair of images, with save_embeddings
# Also has a function to test the input image against the database

img_count = 0
face_count = 0
img_path = ".\\content\\faces_wild"
# for (root, dir, files) in os.walk(img_path):
#     for file in files:
#         path = os.path.join(root, file)
#         faces = extract_face(path)
# if faces is not None:
#     draw_faces(path, faces)

# Print the number of images and faces
# print("Total images: ", img_count)
# print("Total faces: ", face_count)

# Root image paths
root_path = ".\\content\\faces_wild\\lfw-deepfunneled\\lfw-deepfunneled\\"

# Prepare faces_in_the_wild dataset
lfw_allnames = pd.read_csv(".\\content\\faces_wild\\lfw_allnames.csv")
matchpairsDevTest = pd.read_csv(".\\content\\faces_wild\\matchpairsDevTest.csv")
matchpairsDevTrain = pd.read_csv(".\\content\\faces_wild\\matchpairsDevTrain.csv")
mismatchpairsDevTest = pd.read_csv(".\\content\\faces_wild\\mismatchpairsDevTest.csv")
mismatchpairsDevTrain = pd.read_csv(".\\content\\faces_wild\\mismatchpairsDevTrain.csv")
pairs = pd.read_csv(".\\content\\faces_wild\\pairs.csv")
people = pd.read_csv(".\\content\\faces_wild\\people.csv")
peopleDevTest = pd.read_csv(".\\content\\faces_wild\\peopleDevTest.csv")
peopleDevTrain = pd.read_csv(".\\content\\faces_wild\\peopleDevTrain.csv")

# Tidy up the dataset
pairs = pairs.rename(columns={"name": "name1", "Unnamed: 3": "name2"})
matched_pairs = pairs[pairs["name2"].isnull()].drop("name2", axis=1)
mismatched_pairs = pairs[pairs["name2"].notnull()]
mismatched_pairs = mismatched_pairs.rename(columns={"imagenum2": "name2", "name2": "imagenum2"})
# change mismatched_pairs imagenum2 to int
mismatched_pairs["imagenum2"] = mismatched_pairs["imagenum2"].astype(int)
people = people[people.name.notnull()]

# shape data frame
image_paths = lfw_allnames.loc[lfw_allnames.index.repeat(lfw_allnames['images'])]
image_paths['image_path'] = 1 + image_paths.groupby('name').cumcount()
image_paths['image_path'] = image_paths.image_path.apply(lambda x: '{0:0>4}'.format(x))
image_paths['image_path'] = image_paths.name + "/" + image_paths.name + "_" + image_paths.image_path + ".jpg"
image_paths = image_paths.drop("images", 1)

# print(image_paths)

lfw_train, lfw_test = train_test_split(image_paths, test_size=0.2)
lfw_train = lfw_train.reset_index().drop("index", 1)
lfw_test = lfw_test.reset_index().drop("index", 1)

# Format matched pairs and mismatched pairs for testing

matched_pairs["imagenum1"] = root_path + matched_pairs.name1 + "\\" + matched_pairs.name1 + "_" + matched_pairs[
    "imagenum1"].apply(lambda x: '{0:0>4}'.format(x)) + ".jpg"
matched_pairs["imagenum2"] = root_path + matched_pairs.name1 + "\\" + matched_pairs.name1 + "_" + matched_pairs[
    "imagenum2"].apply(lambda x: '{0:0>4}'.format(x)) + ".jpg"
mismatched_pairs["imagenum1"] = root_path + mismatched_pairs.name1 + "\\" + mismatched_pairs.name1 + "_" + \
                                mismatched_pairs["imagenum1"].apply(lambda x: '{0:0>4}'.format(x)) + ".jpg"
mismatched_pairs["imagenum2"] = root_path + mismatched_pairs.name2 + "\\" + mismatched_pairs.name2 + "_" + \
                                mismatched_pairs["imagenum2"].apply(lambda x: '{0:0>4}'.format(x)) + ".jpg"


# Split train and test
# matched_pairs_train, matched_pairs_test = train_test_split(matched_pairs, test_size=0.2)
# mismatched_pairs_train, mismatched_pairs_test = train_test_split(mismatched_pairs, test_size=0.2)


# print("matched_pairs_train: ", matched_pairs_train.shape)
# print("matched_pairs_test: ", matched_pairs_test.shape)
# print("mismatched_pairs_train: ", mismatched_pairs_train.shape)
# print("mismatched_pairs_test: ", mismatched_pairs_test.shape)

threshold = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
threshold_dict = dict()

def matched_pair_evaluate(matched_pairs_test):
    dict_list = []
    for t in threshold:
        cur_threshold_dict = dict()
        # Load the model
        model = vgg_scratch.define_model()
        vgg_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
        detector = mtcnn.MTCNN()
        scores = []
        same = []
        # Get the first 100 pairs of matched pairs
        for i in range(len(matched_pairs_test)):
            filenames = [matched_pairs_test.iloc[i, 1], matched_pairs_test.iloc[i, 2]]
            # print(filenames)
            embeddings = get_embeddings(filenames, detector, vgg_descriptor)
            if embeddings is not None:
                score = is_match(embeddings[0], embeddings[1])
                scores.append(score)
                same.append((lambda a: a <= t)(score))
            else:
                # print("No face detected")
                scores.append(None)
                same.append(None)
        matched_pairs_test["score"] = scores
        matched_pairs_test["same"] = same
        # Save as csv file
        # matched_pairs_test.to_csv("matched_pairs_test_result.csv")
        # Calculate accuracy

        accuracy = len(matched_pairs[matched_pairs["same"] == True]) / (len(matched_pairs[matched_pairs["same"] == True]) + len(matched_pairs[matched_pairs["same"] == False]))
        print("Accuracy: ", accuracy, "Threshold: ", t)
        cur_threshold_dict["threshold"] = t
        cur_threshold_dict["accuracy"] = accuracy

        dict_list.append(cur_threshold_dict)
    threshold_dict["matched_pairs"] = dict_list


# Iterate through matched_pairs_test and extract faces
# matched_pair_evaluate(matched_pairs)


def mismatched_pairs_evaluate(mismatched_pairs_test):
    dict_list = []
    for t in threshold:
        # Load the model
        model = vgg_scratch.define_model()
        vgg_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
        detector = mtcnn.MTCNN()
        scores = []
        same = []
        # Iterate through mismatched_pairs_test and extract faces
        for i in range(len(mismatched_pairs_test)):
            filenames = [mismatched_pairs_test.iloc[i, 1], mismatched_pairs_test.iloc[i, 3]]
            # print(filenames)
            embeddings = get_embeddings(filenames, detector, vgg_descriptor)
            if embeddings is not None:
                score = is_match(embeddings[0], embeddings[1])
                scores.append(score)
                same.append((lambda a: a <= t)(score))
            else:
                # print("No face detected")
                scores.append(None)
                same.append(None)
        mismatched_pairs_test["score"] = scores
        mismatched_pairs_test["same"] = same
        # Save as csv file
        # mismatched_pairs_test.to_csv("mismatched_pairs_test_result.csv")
        # Calculate accuracy
        accuracy = len(mismatched_pairs[mismatched_pairs["same"] == False]) / (len(
            mismatched_pairs[mismatched_pairs["same"] == False] ) + len(
            mismatched_pairs[mismatched_pairs["same"] == True]))
        print(mismatched_pairs_test)
        print("Accuracy: ", accuracy, "Threshold: ", t)
        cur_threshold_dict = dict()
        cur_threshold_dict["threshold"] = t
        cur_threshold_dict["accuracy"] = accuracy

        dict_list.append(cur_threshold_dict)
    threshold_dict["mismatched_pairs"] = dict_list


input_folder = "./content/application_data"
input_path = os.path.join(input_folder, "input_faces")
verified_path_in_appdata = os.path.join(input_folder, "verified_faces")


def evaluate_from_input(mode="avg"):
    input_img_path = os.path.join(input_path, "input.jpg")
    model = define_model()
    vgg_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    detector = mtcnn.MTCNN()
    input_embedding = get_embedding(input_img_path, detector, vgg_descriptor)
    if input_embedding is None:
        raise Exception("No face detected in input image")

    all_distance = {}
    for persons in os.listdir(root_path):
        # print(persons)
        person_distance = []
        images = []
        for image in os.listdir(os.path.join(root_path, persons)):
            full_img_path = os.path.join(root_path, persons, image)
            if full_img_path[-3:] == "jpg":
                images.append(full_img_path)
            # Get embeddings
        embeddings = get_embeddings(images, detector, vgg_descriptor)
        if embeddings is None:
            print("No faces detected")
            continue
        # Check if the input face is a match for the known face
        # print("input_embedding", input_embedding)
        for embedding in embeddings:
            score = is_match(embedding, input_embedding, print_out=True)
            person_distance.append(score)
        # Calculate the average distance for each person
        if mode == "avg":
            all_distance[persons] = np.mean(person_distance)
        elif mode == "min":
            all_distance[persons] = np.min(person_distance)
        else:
            raise Exception("Invalid mode")
        # avg_distance = sum(person_distance) / len(person_distance)
        # all_distance[persons] = avg_distance
    top_ten = sorted(all_distance.items(), key=lambda x: x[1])[:10]
    return top_ten


mismatched_pairs_evaluate(mismatched_pairs)
matched_pair_evaluate(matched_pairs)
print(json.dumps(threshold_dict, indent=4))
with open('comprehensive_result.json', 'w') as f:
    json.dump(threshold_dict, f, indent=2)
# take_photo(input_path)
# top_ten = evaluate_from_input(mode="avg")
# print(top_ten)
# filenames =  [".\\content\\dataset\\img1.jpg", ".\\content\\dataset\\img2.jpg"]
# print(filenames)
# embeddings = get_embeddings(filenames)
# score = is_match(embeddings[0], embeddings[1])
