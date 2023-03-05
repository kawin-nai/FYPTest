import json
import math

import vgg_scratch
from vgg_utils_withsave import *
from vgg_scratch import *

# A file that tests the model on every available pair of images, with save_embeddings
# Also has a function to test the input image against the database

img_count = 0
face_count = 0
img_path = ".\\content\\faces_wild"

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


threshold = (
0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25,
1.3, 1.35, 1.4)
mode = ("l2", "cosine")
# mode = ("cosine",)
threshold_dict = dict()

matched_pairs["actual"] = True
mismatched_pairs["actual"] = False
testing_pairs = pd.concat([matched_pairs, mismatched_pairs])


def matched_pair_evaluate(matched_pairs_test):
    # Load the model
    model = vgg_scratch.define_model()
    vgg_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    detector = mtcnn.MTCNN()
    mode_list = dict()
    for m in mode:
        dict_list = []
        for t in threshold:
            cur_threshold_dict = dict()
            scores = []
            same = []
            # Get the first 100 pairs of matched pairs
            for i in range(len(matched_pairs_test)):
                filenames = [matched_pairs_test.iloc[i, 1], matched_pairs_test.iloc[i, 2]]
                # print(filenames)
                embeddings = get_embeddings(filenames, detector, vgg_descriptor)
                if embeddings is not None:
                    score = is_match(embeddings[0], embeddings[1], mode=m, thresh=t)
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

            accuracy = len(matched_pairs[matched_pairs["same"] == True]) / (
                        len(matched_pairs[matched_pairs["same"] == True]) + len(
                    matched_pairs[matched_pairs["same"] == False]))
            print("Accuracy: ", accuracy, "Threshold: ", t, "Mode: ", m)
            cur_threshold_dict["threshold"] = t
            cur_threshold_dict["accuracy"] = accuracy
            cur_threshold_dict["mode"] = m

            dict_list.append(cur_threshold_dict)
        mode_list[m] = dict_list
    threshold_dict["matched_pairs"] = mode_list


def mismatched_pairs_evaluate(mismatched_pairs_test):
    # Load the model
    model = vgg_scratch.define_model()
    vgg_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    detector = mtcnn.MTCNN()
    mode_list = dict()
    for m in mode:
        dict_list = []
        for t in threshold:
            scores = []
            same = []
            # Iterate through mismatched_pairs_test and extract faces
            for i in range(len(mismatched_pairs_test)):
                filenames = [mismatched_pairs_test.iloc[i, 1], mismatched_pairs_test.iloc[i, 3]]
                # print(filenames)
                embeddings = get_embeddings(filenames, detector, vgg_descriptor)
                if embeddings is not None:
                    score = is_match(embeddings[0], embeddings[1], mode=m, thresh=t)
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
                mismatched_pairs[mismatched_pairs["same"] == False]) + len(
                mismatched_pairs[mismatched_pairs["same"] == True]))
            print(mismatched_pairs_test)
            print("Accuracy: ", accuracy, "Threshold: ", t, "Mode: ", m)
            cur_threshold_dict = dict()
            cur_threshold_dict["threshold"] = t
            cur_threshold_dict["accuracy"] = accuracy
            cur_threshold_dict["mode"] = m

            dict_list.append(cur_threshold_dict)
        mode_list[m] = dict_list
    threshold_dict["mismatched_pairs"] = mode_list


mode_list = dict()
def evaluate(real_pairs):
    # Load the model
    model = vgg_scratch.define_model()
    vgg_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    detector = mtcnn.MTCNN()
    test_pairs = real_pairs
    for m in mode:
        dict_list = []
        for t in threshold:
            scores = []
            same = []
            # Iterate through mismatched_pairs_test and extract faces
            for i in range(len(test_pairs)):
                filenames = [test_pairs.iloc[i, 1], test_pairs.iloc[i, 2]]

                # print(filenames)
                embeddings = get_embeddings(filenames, detector, vgg_descriptor)
                if embeddings is not None:
                    score = is_match(embeddings[0], embeddings[1], mode=m, thresh=t)
                    scores.append(score)
                    same.append((lambda a: a <= t)(score))
                else:
                    # print("No face detected")
                    scores.append(None)
                    same.append(None)
            test_pairs["score"] = scores
            test_pairs["same"] = same

            # Discard the pairs with score = None
            evaluate_pairs = test_pairs[test_pairs["score"].notna()]

            # Calculate metrics
            true_positive = len(evaluate_pairs[(evaluate_pairs["same"] is True) & (evaluate_pairs["same"] == evaluate_pairs["actual"])])
            true_negative = len(evaluate_pairs[(evaluate_pairs["same"] is False) & (evaluate_pairs["same"] == evaluate_pairs["actual"])])
            false_positive = len(evaluate_pairs[(evaluate_pairs["same"] is True) & (evaluate_pairs["same"] != evaluate_pairs["actual"])])
            false_negative = len(evaluate_pairs[(evaluate_pairs["same"] is False) & (evaluate_pairs["same"] != evaluate_pairs["actual"])])

            # Calculate accuracy
            accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f1_score = 2 * precision * recall / (precision + recall)

            # print(test_pairs)
            print("Accuracy: ", accuracy, "Precision: ", precision, "Recall: ", recall, "F1 Score: ", f1_score, "Threshold: ", t, "Mode: ", m)

    #         print("Accuracy: ", accuracy, "Threshold: ", t, "Mode: ", m)
            cur_threshold_dict = dict()
            cur_threshold_dict["mode"] = m
            cur_threshold_dict["threshold"] = t
            cur_threshold_dict["accuracy"] = accuracy
            cur_threshold_dict["precision"] = precision
            cur_threshold_dict["recall"] = recall
            cur_threshold_dict["f1_score"] = f1_score

            dict_list.append(cur_threshold_dict)
        mode_list[m] = dict_list
    # threshold_dict["full_test"] = mode_list


input_folder = "./content/application_data"
input_path = os.path.join(input_folder, "input_faces")
verified_path_in_appdata = os.path.join(input_folder, "verified_faces")


# def evaluate_from_input(mode="avg"):
#     input_img_path = os.path.join(input_path, "input.jpg")
#     model = define_model()
#     vgg_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
#     detector = mtcnn.MTCNN()
#     input_embedding = get_embedding(input_img_path, detector, vgg_descriptor)
#     if input_embedding is None:
#         raise Exception("No face detected in input image")
#
#     all_distance = {}
#     for persons in os.listdir(root_path):
#         # print(persons)
#         person_distance = []
#         images = []
#         for image in os.listdir(os.path.join(root_path, persons)):
#             full_img_path = os.path.join(root_path, persons, image)
#             if full_img_path[-3:] == "jpg":
#                 images.append(full_img_path)
#             # Get embeddings
#         embeddings = get_embeddings(images, detector, vgg_descriptor)
#         if embeddings is None:
#             print("No faces detected")
#             continue
#         # Check if the input face is a match for the known face
#         # print("input_embedding", input_embedding)
#         for embedding in embeddings:
#             score = is_match(embedding, input_embedding, print_out=True)
#             person_distance.append(score)
#         # Calculate the average distance for each person
#         if mode == "avg":
#             all_distance[persons] = np.mean(person_distance)
#         elif mode == "min":
#             all_distance[persons] = np.min(person_distance)
#         else:
#             raise Exception("Invalid mode")
#         # avg_distance = sum(person_distance) / len(person_distance)
#         # all_distance[persons] = avg_distance
#     top_ten = sorted(all_distance.items(), key=lambda x: x[1])[:10]
#     return top_ten


# mismatched_pairs_evaluate(mismatched_pairs)
# matched_pair_evaluate(matched_pairs)
# evaluate(testing_pairs)
# print(json.dumps(threshold_dict, indent=4))
# with open('comprehensive_result_all.json', 'w') as f:
#     json.dump(mode_list, f, indent=2)
# take_photo(input_path)
# top_ten = evaluate_from_input(mode="avg")
# print(top_ten)
# filenames =  [".\\content\\dataset\\img1.jpg", ".\\content\\dataset\\img2.jpg"]
# print(filenames)
# embeddings = get_embeddings(filenames)
# score = is_match(embeddings[0], embeddings[1])

with open('comprehensive_result_all.json', 'r') as f:
    data = json.load(f)

    # l2
    l2_list = data["l2"]
    threshold_list = []
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    for i in l2_list:
        threshold_list.append(i["threshold"])
        accuracy_list.append(i["accuracy"])
        precision_list.append(i["precision"])
        recall_list.append(i["recall"])
        f1_score_list.append(i["f1_score"])

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.xticks(np.arange(0.1, 1.5, 0.2))
    plt.plot(threshold_list, accuracy_list, label="l2 accuracy")
    plt.title("accuracy")

    plt.subplot(2, 2, 2)
    plt.xticks(np.arange(0.1, 1.5, 0.2))
    plt.plot(threshold_list, precision_list, label="l2 precision")
    plt.title("precision")

    plt.subplot(2, 2, 3)
    plt.xticks(np.arange(0.1, 1.5, 0.2))
    plt.plot(threshold_list, recall_list, label="l2 recall")
    plt.title("recall")

    plt.subplot(2, 2, 4)
    plt.xticks(np.arange(0.1, 1.5, 0.2))
    plt.plot(threshold_list, f1_score_list, label="l2 f1_score")
    plt.title("f1_score")

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.show()

    # cosine
    cosine_list = data["cosine"]
    threshold_list = []
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []

    for i in cosine_list:
        threshold_list.append(i["threshold"])
        accuracy_list.append(i["accuracy"])
        precision_list.append(i["precision"])
        recall_list.append(i["recall"])
        f1_score_list.append(i["f1_score"])

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.xticks(np.arange(0.1, 1.5, 0.2))
    plt.plot(threshold_list, accuracy_list, label="cosine accuracy")
    plt.title("accuracy")

    plt.subplot(2, 2, 2)
    plt.xticks(np.arange(0.1, 1.5, 0.2))
    plt.plot(threshold_list, precision_list, label="cosine precision")
    plt.title("precision")

    plt.subplot(2, 2, 3)
    plt.xticks(np.arange(0.1, 1.5, 0.2))
    plt.plot(threshold_list, recall_list, label="cosine recall")
    plt.title("recall")

    plt.subplot(2, 2, 4)
    plt.xticks(np.arange(0.1, 1.5, 0.2))
    plt.plot(threshold_list, f1_score_list, label="cosine f1_score")
    plt.title("f1_score")

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.show()

    top_three_l2 = dict()
    top_three_cosine = dict()
    for mode in data:
        if mode == "l2":
            top_three_l2["accuracy"] = sorted(data[mode], key=lambda x: x["accuracy"], reverse=True)[:3]
            top_three_l2["precision"] = sorted(data[mode], key=lambda x: x["precision"], reverse=True)[:3]
            top_three_l2["recall"] = sorted(data[mode], key=lambda x: x["recall"], reverse=True)[:3]
            top_three_l2["f1_score"] = sorted(data[mode], key=lambda x: x["f1_score"], reverse=True)[:3]
        else:
            top_three_cosine["accuracy"] = sorted(data[mode], key=lambda x: x["accuracy"], reverse=True)[:3]
            top_three_cosine["precision"] = sorted(data[mode], key=lambda x: x["precision"], reverse=True)[:3]
            top_three_cosine["recall"] = sorted(data[mode], key=lambda x: x["recall"], reverse=True)[:3]
            top_three_cosine["f1_score"] = sorted(data[mode], key=lambda x: x["f1_score"], reverse=True)[:3]
    print(top_three_l2)
    print(top_three_cosine)

    with open('top_three_l2.json', 'w') as ff:
        json.dump(top_three_l2, ff, indent=2)
    with open('top_three_cosine.json', 'w') as ff:
        json.dump(top_three_cosine, ff, indent=2)