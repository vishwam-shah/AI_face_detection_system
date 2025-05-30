# import tkinter as tk
# from tkinter import messagebox
# import cv2
# import os
# import csv
# import numpy as np
# from PIL import Image
# import pandas as pd
# from pathlib import Path
# import threading
# import time

# # Global variable to control the thread
# is_tracking = False

# # Set up paths
# base_dir = Path(__file__).resolve().parent
# data_dir = base_dir / "data"
# training_image_dir = base_dir / "TrainingImage"
# user_details_file = base_dir / "UserDetails" / "UserDetails.csv"
# trainer_file = training_image_dir / "Trainer.yml"
# unknown_image_dir = base_dir / "UnknownImages"  # Directory for unknown images

# # Ensure directories exist
# training_image_dir.mkdir(parents=True, exist_ok=True)
# unknown_image_dir.mkdir(parents=True, exist_ok=True)
# user_details_file.parent.mkdir(parents=True, exist_ok=True)

# # Load Haarcascade for face detection
# harcascadePath = str(data_dir / "haarcascade_frontalface_default.xml")

# # Main window setup
# window = tk.Tk()
# window.title("Face Recognizer")
# window.configure(background='white')

# message = tk.Label(
#     window, text="Face Recognition System",
#     bg="green", fg="white", width=50,
#     height=3, font=('times', 30, 'bold'))
# message.place(x=200, y=20)

# # Input fields
# def create_label_and_entry(text, x, y):
#     label = tk.Label(window, text=text, width=20, fg="green",
#                      bg="white", height=2, font=('times', 15, 'bold'))
#     label.place(x=x, y=y)
#     entry = tk.Entry(window, width=20, bg="white",
#                      fg="green", font=('times', 15, 'bold'))
#     entry.place(x=x + 300, y=y + 15)
#     return entry

# txt = create_label_and_entry("No.", 400, 200)
# txt2 = create_label_and_entry("Name", 400, 300)

# def is_number(s):
#     try:
#         float(s)
#         return True
#     except ValueError:
#         return False

# def TakeImages():
#     Id = txt.get()
#     name = txt2.get()

#     if not is_number(Id) or not name.isalpha():
#         res = "Enter valid ID (numeric) and Name (alphabetical)"
#         message.configure(text=res)
#         return

#     cam = cv2.VideoCapture(0)
#     detector = cv2.CascadeClassifier(harcascadePath)
    
#     if detector.empty():
#         messagebox.showerror("Error", "Haar cascade file not found.")
#         return
    
#     sampleNum = 0

#     while True:
#         ret, img = cam.read()
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = detector.detectMultiScale(gray, 1.3, 5)

#         for (x, y, w, h) in faces:
#             sampleNum += 1
#             cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             cv2.imwrite(str(training_image_dir / f"{name}.{Id}.{sampleNum}.jpg"), gray[y:y + h, x:x + w])
#             cv2.imshow('frame', img)

#         if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum >= 60:
#             break

#     cam.release()
#     cv2.destroyAllWindows()
    
#     # Write to CSV with header if the file is empty
#     with open(user_details_file, 'a+', newline='') as csvFile:
#         writer = csv.writer(csvFile)
#         if csvFile.tell() == 0:  # Check if file is empty
#             writer.writerow(['Id', 'Name'])  # Write header if empty
#         writer.writerow([Id, name])
    
#     message.configure(text=f"Images Saved for ID: {Id}, Name: {name}")

# def TrainImages():
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     detector = cv2.CascadeClassifier(harcascadePath)
#     faces, Ids = getImagesAndLabels(str(training_image_dir))
    
#     if len(faces) == 0:
#         messagebox.showerror("Error", "No training images found.")
#         return

#     recognizer.train(faces, np.array(Ids))
#     recognizer.save(str(trainer_file))
#     print(f"Model saved at: {trainer_file}")  # Debug output
#     messagebox.showinfo("Success", "Model Trained Successfully")

# def getImagesAndLabels(path):
#     imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]
#     faces = []
#     Ids = []
#     for imagePath in imagePaths:
#         pilImage = Image.open(imagePath).convert('L')
#         imageNp = np.array(pilImage, 'uint8')
#         Id = int(os.path.split(imagePath)[-1].split(".")[1])
#         faces.append(imageNp)
#         Ids.append(Id)
#     return faces, Ids

# def TrackImages():
#     global is_tracking
#     is_tracking = True

#     def start_tracking():
#         global is_tracking  # Ensure to use the global variable
#         recognizer = cv2.face.LBPHFaceRecognizer_create()
        
#         # Ensure trainer file exists before trying to read
#         if not os.path.exists(str(trainer_file)):
#             messagebox.showerror("Error", "Trainer file not found. Please train the model first.")
#             return
        
#         recognizer.read(str(trainer_file))
#         faceCascade = cv2.CascadeClassifier(harcascadePath)
#         df = pd.read_csv(user_details_file)

#         cam = cv2.VideoCapture(0)
#         font = cv2.FONT_HERSHEY_SIMPLEX

#         messagebox.showinfo("Testing", "Face Recognition Testing Started")

#         start_time = time.time()  # Record the start time
#         duration = 60  # Duration in seconds

#         while is_tracking and (time.time() - start_time < duration):
#             ret, im = cam.read()
#             if not ret:
#                 messagebox.showerror("Error", "Failed to capture video.")
#                 break

#             gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#             faces = faceCascade.detectMultiScale(gray, 1.2, 5)

#             for (x, y, w, h) in faces:
#                 cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
#                 Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
#                 name_display = 'Unknown'

#                 # Safeguard to avoid KeyError
#                 if Id in df['Id'].values:
#                     name_display = df.loc[df['Id'] == Id]['Name'].values[0]

#                 cv2.putText(im, name_display, (x, y + h), font, 1, (255, 255, 255), 2)

#             cv2.imshow('Face Recognition', im)

#             if cv2.waitKey(1) == ord('q'):
#                 break

#         is_tracking = False  # Stop tracking after the duration
#         cam.release()
#         cv2.destroyAllWindows()
#         messagebox.showinfo("Testing", "Face Recognition Testing Ended")

#     tracking_thread = threading.Thread(target=start_tracking)
#     tracking_thread.start()


# # Buttons
# buttons = [
#     ("Sample", TakeImages, 200),
#     ("Training", TrainImages, 500),
#     ("Testing", TrackImages, 800),
#     ("Quit", window.destroy, 1100)
# ]

# for text, command, x in buttons:
#     tk.Button(window, text=text, command=command, fg="white", bg="green", 
#               width=20, height=3, activebackground="Red", 
#               font=('times', 15, 'bold')).place(x=x, y=500)

# window.mainloop()



# FACE NET IMPLEMENTED BELOW...

import tkinter as tk
from tkinter import messagebox
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
import threading
import time
from facenet_pytorch import InceptionResnetV1
import torch
from torchvision import transforms

# Global variable to control the thread
is_tracking = False

# Set up paths
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"
training_image_dir = base_dir / "TrainingImage"
user_details_file = base_dir / "UserDetails" / "UserDetails.csv"
embeddings_file = base_dir / "UserDetails" / "face_embeddings.csv"
unknown_image_dir = base_dir / "UnknownImages"  # Directory for unknown images

# Ensure directories exist
training_image_dir.mkdir(parents=True, exist_ok=True)
unknown_image_dir.mkdir(parents=True, exist_ok=True)
user_details_file.parent.mkdir(parents=True, exist_ok=True)

# Load Haarcascade for face detection
harcascadePath = str(data_dir / "haarcascade_frontalface_default.xml")

# Load FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Main window setup
window = tk.Tk()
window.title("Face Recognizer")
window.configure(background='white')

message = tk.Label(
    window, text="Face Recognition System",
    bg="green", fg="white", width=50,
    height=3, font=('times', 30, 'bold'))
message.place(x=200, y=20)

# Input fields
def create_label_and_entry(text, x, y):
    label = tk.Label(window, text=text, width=20, fg="green",
                     bg="white", height=2, font=('times', 15, 'bold'))
    label.place(x=x, y=y)
    entry = tk.Entry(window, width=20, bg="white",
                     fg="green", font=('times', 15, 'bold'))
    entry.place(x=x + 300, y=y + 15)
    return entry

txt = create_label_and_entry("No.", 400, 200)
txt2 = create_label_and_entry("Name", 400, 300)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Define a transformation pipeline for FaceNet
face_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def TakeImages():
    Id = txt.get()
    name = txt2.get()

    if not is_number(Id) or not name.isalpha():
        res = "Enter valid ID (numeric) and Name (alphabetical)"
        message.configure(text=res)
        return

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(harcascadePath)
    
    if detector.empty():
        messagebox.showerror("Error", "Haar cascade file not found.")
        return
    
    sampleNum = 0
    embeddings = []

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sampleNum += 1
            face = img[y:y + h, x:x + w]
            face_pil = Image.fromarray(face)
            
            # Transform the face for FaceNet
            face_tensor = face_transform(face_pil).unsqueeze(0)
            with torch.no_grad():
                embedding = model(face_tensor).numpy()[0]
            embeddings.append((Id, name, *embedding))
            cv2.imshow('frame', img)

        if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum >= 20:
            break

    cam.release()
    cv2.destroyAllWindows()
    
    # Save embeddings to CSV
    with open(embeddings_file, 'a+', newline='') as csvFile:
        writer = csv.writer(csvFile)
        if csvFile.tell() == 0:  # Write header if empty
            header = ['Id', 'Name'] + [f'Embed_{i}' for i in range(len(embedding))]
            writer.writerow(header)
        writer.writerows(embeddings)
    
    message.configure(text=f"Images and embeddings saved for ID: {Id}, Name: {name}")

def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def TrackImages():
    global is_tracking
    is_tracking = True

    def start_tracking():
        global is_tracking
        faceCascade = cv2.CascadeClassifier(harcascadePath)
        known_embeddings = pd.read_csv(embeddings_file)

        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        messagebox.showinfo("Testing", "Face Recognition Testing Started")
        start_time = time.time()  # Record the start time
        duration = 60  # Duration in seconds

        while is_tracking and (time.time() - start_time < duration):
            ret, im = cam.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture video.")
                break

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
                face = im[y:y + h, x:x + w]
                face_pil = Image.fromarray(face)
                
                # Transform face for FaceNet
                face_tensor = face_transform(face_pil).unsqueeze(0)
                with torch.no_grad():
                    current_embedding = model(face_tensor).numpy()[0]

                # Find closest match
                best_match = None
                best_score = 0.6  # Threshold for cosine similarity
                for i, row in known_embeddings.iterrows():
                    embedding = np.array(row[2:], dtype=np.float32)
                    similarity = cosine_similarity(current_embedding, embedding)
                    if similarity > best_score:
                        best_score = similarity
                        best_match = row['Name']
                
                name_display = best_match if best_match else 'Unknown'
                cv2.putText(im, name_display, (x, y + h), font, 1, (255, 255, 255), 2)

            cv2.imshow('Face Recognition', im)

            if cv2.waitKey(1) == ord('q'):
                break

        is_tracking = False  # Stop tracking after the duration
        cam.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Testing", "Face Recognition Testing Ended")

    tracking_thread = threading.Thread(target=start_tracking)
    tracking_thread.start()

# Buttons
buttons = [
    ("Sample", TakeImages, 200),
    ("Training", lambda: messagebox.showinfo("Info", "Training is embedded within the Take Images step."), 500),
    ("Testing", TrackImages, 800),
    ("Quit", window.destroy, 1100)
]

for text, command, x in buttons:
    tk.Button(window, text=text, command=command, fg="white", bg="green", 
              width=20, height=3, activebackground="Red", 
              font=('times', 15, 'bold')).place(x=x, y=500)

window.mainloop()
