import os
import cv2
import time

def take_photo(name):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow('Webcam', frame)
        # Press 'q' to capture the image
        if cv2.waitKey(1) & 0xFF == ord('q'):
            timestamp = str(int(time.time()))
            break
    try:
        if ret:
            cv2.imwrite(os.path.join(os.path.join(verified_path, name), name+"_" + timestamp+".jpg"), frame)
            print("Input picture taken")
    except:
        print("Error taking input picture")
    cap.release()
    cv2.destroyAllWindows()

# Get timestamp


img_path = "./content/application_data"
input_path = os.path.join(img_path, "input_faces")
verified_path = os.path.join(img_path, "verified_faces")
# Input name from console input
name = input("Enter your full name (Last name first): ")
# Replace spaces with underscores
name = name.replace(" ", "_")
# Create a folder for the input name if it doesn't exist
if not os.path.exists(os.path.join(verified_path, name)):
    os.makedirs(os.path.join(verified_path, name))
# Take a photo
take_photo(name)

