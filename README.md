# AI Face Recognition Attendance System

A GUI-based Face Recognition System designed for **automated attendance marking** using OpenCV. This system uses **frontal face detection**, grayscale image conversion for enhanced accuracy, and consists of **three core components**: Sample Collection, Training, and Testing (Recognition).

---

## 💡 Features

- 🎦 Real-time face detection using OpenCV's Haar Cascade.
- 🧑‍💻 GUI-based interface for user-friendly interaction.
- 📸 Image samples are converted to **grayscale** for optimal training accuracy.
- 🏋️‍♂️ Model training using **LBPH Face Recognizer**.
- ✅ Attendance marking upon successful recognition.
- 📁 Auto-creation of dataset and training model folders.

---

## 📁 Components

### 1. Sample Collection
- Captures face samples using a webcam.
- Stores grayscale images in the `dataSet` directory.
- Requires a unique User ID and Name input via GUI.

### 2. Model Training
- Uses the LBPH algorithm to train the face recognizer.
- Processes all grayscale samples stored in the `dataSet`.
- Trained model saved as `trainer.yml` in the `trainer` directory.

### 3. Face Recognition (Testing)
- Loads the trained model (`trainer.yml`) to detect and recognize faces.
- Displays real-time camera feed.
- Marks attendance in `attendance.csv` upon successful face match.

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python | Programming Language |
| OpenCV | Face Detection and Recognition |
| Tkinter | GUI Framework |
| NumPy | Numerical operations |
| CSV | Attendance data storage |

---

## 🖥️ GUI Overview

| Section | Function |
|--------|----------|
| Sample | Capture face samples for a new user |
| Train  | Train the recognizer with all samples |
| Test   | Detect and recognize faces for attendance |

---

## 📦 Directory Structure


---

## 🚀 How to Run

1. **Install Dependencies**

```bash
pip install opencv-python numpy
```

2. **Run**
```bash 
python main.py
```
3. **Use GUI Buttons**

Click Sample to register a new user.

Click Train to train the recognizer.

Click Test to detect and mark attendance.
