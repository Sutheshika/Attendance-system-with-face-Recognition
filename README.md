# Attendance System with Face Recognition

A complete end-to-end face recognition system designed for automated attendance tracking. This project uses deep learning and machine learning techniques to detect, recognize, and identify individuals from video streams or images.

## Project Overview

This system performs facial recognition and identification using modern computer vision techniques. It captures face data, processes facial embeddings, trains an ML model, and finally recognizes individuals in real-time.

### Key Features
- Real-time face detection and recognition
- Student/Person dataset creation with webcam
- Facial embedding extraction using OpenFace
- SVM-based face classification
- CSV database for storing student information
- Multi-stage pipeline approach

---

## Environment Setup

### Prerequisites
- **Python 3.7+**
- **Windows/Linux/Mac** Operating System
- **Webcam** (for dataset creation and real-time recognition)

### Required Libraries

Install all dependencies using:

```bash
pip install opencv-python imutils numpy scikit-learn scikit-image dlib
```

Or install from requirements (if available):

```bash
pip install -r requirements.txt
```

### Pre-trained Models

The following pre-trained models are included in the project:

1. **haarcascade_frontalface_default.xml**
   - Haar Cascade classifier for face detection
   - Located in the project root directory

2. **openface_nn4.small2.v1.t7**
   - PyTorch model for extracting 128-dimensional face embeddings
   - Used for facial feature extraction

3. **model/deploy.prototxt** & **model/res10_300x300_ssd_iter_140000.caffemodel**
   - Caffe-based deep learning models for robust face detection
   - More accurate than Haar Cascades

### Project Structure

```
Attendance-system-with-face-Recognition/
├── 1_datasetCreation.py              # Step 1: Capture face images
├── 2_preprocessingEmbeddings.py       # Step 2: Extract facial embeddings
├── 3_trainingFaceML.py               # Step 3: Train SVM classifier
├── 4_recognizingPerson.py            # Step 4: Real-time recognition
├── 5_recognizingPersonwithCSVDatabase.py  # Step 5: Recognition with CSV logging
├── haarcascade_frontalface_default.xml
├── openface_nn4.small2.v1.t7
├── student.csv                        # Student database (name, roll number)
├── model/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
└── output/
    ├── embeddings.pickle              # Extracted face embeddings
    ├── recognizer.pickle              # Trained SVM model
    └── le.pickle                      # Label encoder
```

---

## How to Use This Project

This is a **5-step pipeline** that must be executed sequentially:

### Step 1: Dataset Creation

**Script:** `1_datasetCreation.py`

This script captures 50 face images of a person using the webcam.

**How to Run:**
```bash
python 1_datasetCreation.py
```

**What It Does:**
- Prompts for student name and roll number
- Creates a directory with the student's name in the `dataset/` folder
- Captures 50 face images using the webcam
- Stores images in the format: `00000.png`, `00001.png`, etc.
- Records student information in `student.csv`

**User Input Example:**
```
Enter your Name: John Doe
Enter your Roll_Number: 101
```

**Instructions:**
- Allow proper lighting
- Position face clearly in frame
- Press `Q` to exit early
- Repeat for each student in the system

---

### Step 2: Preprocessing & Facial Embeddings

**Script:** `2_preprocessingEmbeddings.py`

This script processes all captured images and extracts 128-dimensional facial embeddings.

**How to Run:**
```bash
python 2_preprocessingEmbeddings.py
```

**What It Does:**
- Reads all images from the `dataset/` folder
- Detects faces using the Caffe deep learning model
- Extracts facial embeddings using OpenFace model
- Saves embeddings to `output/embeddings.pickle`
- Creates mappings between embeddings and student names

**Output:**
- `output/embeddings.pickle` - Contains all facial embeddings and corresponding names

---

### Step 3: Training the ML Model

**Script:** `3_trainingFaceML.py`

This script trains an SVM (Support Vector Machine) classifier on the extracted embeddings.

**How to Run:**
```bash
python 3_trainingFaceML.py
```

**What It Does:**
- Loads extracted embeddings from `output/embeddings.pickle`
- Encodes student names into numeric labels
- Trains an SVM classifier with linear kernel
- Saves the trained model to `output/recognizer.pickle`
- Saves the label encoder to `output/le.pickle`

**Output Files:**
- `output/recognizer.pickle` - Trained SVM model
- `output/le.pickle` - Label encoder for decoding predictions

---

### Step 4: Real-Time Face Recognition

**Script:** `4_recognizingPerson.py`

This script performs real-time face recognition using the webcam.

**How to Run:**
```bash
python 4_recognizingPerson.py
```

**What It Does:**
- Loads the trained SVM model and embeddings
- Accesses the webcam for live video stream
- Detects faces in real-time
- Predicts identity for each detected face
- Displays results with bounding boxes and confidence scores
- Shows predictions on-screen

**Controls:**
- Press `Q` to quit the program

**Output:**
- Real-time video with:
  - Green/colored bounding boxes around detected faces
  - Student name overlaid on detected faces
  - Confidence score (probability) of the prediction

---

### Step 5: Recognition with CSV Database Logging

**Script:** `5_recognizingPersonwithCSVDatabase.py`

This is an enhanced version that logs recognized faces to a CSV file (for attendance tracking).

**How to Run:**
```bash
python 5_recognizingPersonwithCSVDatabase.py
```

**What It Does:**
- Performs real-time face recognition (same as Step 4)
- Logs each recognized person to an attendance file
- Retrieves student roll number from `student.csv`
- Records timestamp and recognition details
- Useful for automated attendance marking

**Output:**
- Real-time video display with recognition
- Attendance records saved to a CSV file
- Roll number and timestamp logging

**Controls:**
- Press `Q` to quit

---

## Complete Workflow

### First Time Setup:

```bash
# Step 1: Create dataset for each student
python 1_datasetCreation.py  # Repeat for each student

# Step 2: Extract facial embeddings
python 2_preprocessingEmbeddings.py

# Step 3: Train the model
python 3_trainingFaceML.py
```

### For Daily Use (Attendance):

```bash
# Use either:
# Option A - Basic recognition (display only)
python 4_recognizingPerson.py

# Option B - Recognition with attendance logging
python 5_recognizingPersonwithCSVDatabase.py
```

---

## Important Notes

### Before Running:

1. **Ensure Webcam Access:**
   - Check that your webcam is connected and accessible
   - Grant camera permissions if prompted

2. **Dataset Directory:**
   - Create a `dataset/` folder in the project root if it doesn't exist
   - The system will create it automatically if needed

3. **Model Files:**
   - Ensure all model files exist in the `model/` directory
   - Verify `openface_nn4.small2.v1.t7` is in the root directory

4. **CSV File:**
   - `student.csv` is created automatically during dataset creation
   - Format: `Name, Roll_Number`

### Troubleshooting:

| Problem | Solution |
|---------|----------|
| Webcam not detected | Check permissions and device connection |
| "File not found" error | Ensure all model files are in correct paths |
| Poor recognition accuracy | Capture more face images (50+ per person) with varied angles and lighting |
| Slow processing | Reduce video frame resolution or increase detection confidence threshold |

---

## System Performance Tips

1. **Lighting:** Use consistent, well-lit environments for better results
2. **Face Angle:** Capture faces at various angles (frontal, slight left/right)
3. **Distance:** Maintain consistent distance from camera (0.5-2 meters)
4. **Dataset Size:** More images (100-200) yield better accuracy
5. **Model Selection:** Deep learning model (Caffe) is more accurate than Haar Cascades

---

## Technologies Used

- **OpenCV (cv2)** - Computer vision library
- **imutils** - Image processing utilities
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning (SVM, Label Encoding)
- **PyTorch/Torch** - OpenFace model for embeddings
- **Caffe** - Deep learning framework for face detection

---

## Output Files Generated

After running the pipeline, the following files will be created:

- `dataset/` - Directory containing face images organized by person
- `output/embeddings.pickle` - Serialized facial embeddings
- `output/recognizer.pickle` - Trained SVM classifier
- `output/le.pickle` - Label encoder for predictions
- `student.csv` - Student database with names and roll numbers
- Attendance logs (optional) - Generated by script 5

---

## License

This project is for educational purposes.

---

## Support

For issues or questions, ensure all steps are followed in sequence and check the troubleshooting section above.

