# train_model.py

import os
import cv2
import numpy as np
import pandas as pd
from utils.face_utils import get_face_detector, detect_faces

# --------- 🔧 Paths ---------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CSV_PATH = os.path.join(BASE_DIR, "patient_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "trained_model", "face_model.yml")

# Ensure trained_model folder exists
os.makedirs(os.path.join(BASE_DIR, "trained_model"), exist_ok=True)

def get_images_and_labels():
    image_paths = []
    labels = []
    label_map = {}
    reverse_map = {}
    label_count = 0

    for patient_id in os.listdir(DATASET_DIR):
        patient_dir = os.path.join(DATASET_DIR, patient_id)
        if not os.path.isdir(patient_dir):
            continue

        for img_file in os.listdir(patient_dir):
            img_path = os.path.join(patient_dir, img_file)
            if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Assign label to patient ID
            if patient_id not in label_map:
                label_map[patient_id] = label_count
                reverse_map[label_count] = patient_id
                label_count += 1

            image_paths.append(img)
            labels.append(label_map[patient_id])

    return image_paths, np.array(labels), reverse_map

def train_and_save_model():
    images, labels, reverse_map = get_images_and_labels()

    if len(images) == 0:
        print("[ERROR] No training data found.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, labels)
    recognizer.save(MODEL_PATH)

    # Save label mappings to CSV for lookup later
    pd.DataFrame.from_dict(reverse_map, orient='index').reset_index().rename(columns={"index": "Label", 0: "PatientID"}).to_csv("label_map.csv", index=False)
    print("[INFO] Model trained and saved to:", MODEL_PATH)

if __name__ == "__main__":
    train_and_save_model()
