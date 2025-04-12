import cv2
import os
import numpy as np
import pandas as pd

# Paths
MODEL_PATH = os.path.join("trained_model", "face_model.yml")
LABEL_MAP_PATH = "label_map.csv"
PATIENT_DATA_PATH = "patient_data.csv"
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Load face recognizer and cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Load label map
label_df = pd.read_csv(LABEL_MAP_PATH)
label_map = dict(zip(label_df["Label"], label_df["PatientID"]))

# Load patient data
patient_df = pd.read_csv(PATIENT_DATA_PATH)
patient_data_map = dict(zip(patient_df["PatientID"], patient_df["Name"]))

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))

        label, confidence = recognizer.predict(face_img)
        print(f"[DEBUG] Label: {label}, Confidence: {confidence}")

        if confidence < 60:  # Lower = better match
            patient_id = label_map.get(label, "Unknown")
            name = patient_data_map.get(patient_id, "Unknown")
            text = f"{name} ({patient_id})"
        else:
            text = "Unknown"

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Patient Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
