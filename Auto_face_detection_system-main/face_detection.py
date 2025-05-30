import os
import cv2
import face_recognition
import numpy as np
from sklearn import neighbors
import pickle
from PIL import Image  # To handle image conversion

# Function to train the KNN classifier with known faces
def train_knn(known_face_encodings, known_face_names, model_save_path="knn_model.clf"):
    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree', weights='distance')
    knn_clf.fit(known_face_encodings, known_face_names)

    # Save the trained KNN model
    with open(model_save_path, 'wb') as f:
        pickle.dump(knn_clf, f)

    return knn_clf

# Function to load known faces and train the KNN classifier
def load_known_faces(known_faces_dir=r"C:\Users\Owner\Desktop\me"):
    known_face_encodings = []
    known_face_names = []

    for file_name in os.listdir(known_faces_dir):
        if file_name.endswith('.jpeg') or file_name.endswith('.png'):
            # Extract the person's name from the filename
            name = os.path.splitext(file_name)[0]

            # Load the image file
            image_path = os.path.join(known_faces_dir, file_name)
            image = face_recognition.load_image_file(image_path)

            # Convert the image to RGB if it's not in RGB format
            if image.ndim == 2:  # If it's a grayscale image
                image = np.stack((image,) * 3, axis=-1)  # Convert grayscale to RGB
            elif image.shape[2] != 3:  # If it's not RGB (3 channels)
                image = Image.open(image_path).convert("RGB")  # Use PIL to force convert to RGB
                image = np.array(image)  # Convert back to numpy array

            # Get the face encoding
            face_encodings = face_recognition.face_encodings(image)
            if len(face_encodings) > 0:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(name)

    return known_face_encodings, known_face_names

# Load or train the KNN model
def load_knn_model(model_path="knn_model.clf"):
    # Load the KNN classifier from the saved file
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)
    return knn_clf

# Function to recognize faces in the frame using the KNN classifier
def recognize_faces_in_frame(knn_clf, frame):
    # Convert the frame to RGB format (OpenCV uses BGR by default)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    names = []

    # Ensure we have valid face encodings before proceeding
    if len(face_encodings) > 0:
        # Use the KNN model to predict the name of the person in each face
        closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
        is_recognized = [closest[0] <= 0.4 for closest in closest_distances[0]]
        
        predictions = knn_clf.predict(face_encodings)

        for pred, rec in zip(predictions, is_recognized):
            if rec:
                names.append(pred)
            else:
                names.append("Unknown")

    return face_locations, names

# Main function to capture video, detect faces, and recognize them
def main():
    known_face_encodings, known_face_names = load_known_faces()

    # Train KNN model or load a pre-trained model if available
    if not os.path.exists("knn_model.clf"):
        knn_clf = train_knn(known_face_encodings, known_face_names)
    else:
        knn_clf = load_knn_model()

    # Initialize the webcam video stream
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Failed to capture video")
            break

        # Recognize faces in the current frame
        face_locations, face_names = recognize_faces_in_frame(knn_clf, frame)

        # Draw rectangles and names around the faces
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow('Face Recognition', frame)

        # Press 'q' to quit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
