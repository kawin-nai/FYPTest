from vgg_utils import *
import cv2

img_path = "./content/application_data"
input_path = os.path.join(img_path, "input_faces")
verified_path = os.path.join(img_path, "verified_faces")


# Open webcam and take a input picture using OpenCV
def take_photo():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow('Webcam', frame)
        # Press 'q' to capture the image
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    try:
        if ret:
            cv2.imwrite(os.path.join(input_path, "input.jpg"), frame)
            print("Input picture taken")
    except:
        print("Error taking input picture")
    cap.release()
    cv2.destroyAllWindows()


take_photo()

input_img_path = os.path.join(input_path, "input.jpg")
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
detector = mtcnn.MTCNN()

all_distance = {}
for persons in os.listdir(verified_path):
    # print(persons)
    person_distance = []
    for images in os.listdir(os.path.join(verified_path, persons)):
        full_img_path = os.path.join(verified_path, persons, images)
        images = [full_img_path, input_img_path]
        # Get embeddings
        embeddings = get_embeddings(images, detector, model)
        if embeddings is None:
            print("No face detected")
            continue
        # Check if the input face is a match for the known face
        score = is_match(embeddings[0], embeddings[1])
        person_distance.append(score)
    # Calculate the average distance for each person
    avg_distance = sum(person_distance) / len(person_distance)
    all_distance[persons] = avg_distance

# Get the top three persons with the lowest distance
top_three = sorted(all_distance.items(), key=lambda x: x[1])[:3]
print(top_three)
