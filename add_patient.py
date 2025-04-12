# add_patient.py

import cv2
import os
import pandas as pd
from utils.face_utils import get_face_detector, detect_faces

# --------- 🔧 Path Setup ---------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CSV_PATH = os.path.join(BASE_DIR, "patient_data.csv")

# Ensure dataset folder exists
os.makedirs(DATASET_DIR, exist_ok=True)

# Create CSV if it doesn't exist
if not os.path.exists(CSV_PATH):
    df = pd.DataFrame(columns=["PatientID", "Name"])
    df.to_csv(CSV_PATH, index=False)

# --------- 📸 Capture Faces ---------
def capture_patient_faces(patient_id, name):
    face_cascade = get_face_detector()
    cam = cv2.VideoCapture(0)
    count = 0
    patient_dir = os.path.join(DATASET_DIR, str(patient_id))

    # Create folder for this patient's images
    os.makedirs(patient_dir, exist_ok=True)

    print("[INFO] Capturing face images. Look at the camera...")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Failed to capture frame from camera.")
            break

        faces, gray = detect_faces(frame, face_cascade)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            img_path = os.path.join(patient_dir, f"{count}.jpg")
            cv2.imwrite(img_path, face_img)
            count += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Image {count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Capturing Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:
            break

    cam.release()
    cv2.destroyAllWindows()

    # Save to CSV
    df = pd.read_csv(CSV_PATH)
    df = pd.concat([df, pd.DataFrame([[patient_id, name]], columns=["PatientID", "Name"])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"[INFO] Captured {count} images and saved patient data.")

# --------- 🚀 Run Script ---------
if __name__ == "__main__":
    name = input("Enter patient name: ")
    patient_id = input("Enter patient ID (must be unique): ")
    capture_patient_faces(patient_id, name)
