from vgg_utils import *


# Root image paths
root_path = "./content/faces_wild/lfw-deepfunneled_masked/"

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
