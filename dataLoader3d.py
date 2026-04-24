import os
import numpy as np
import pandas as pd
import pydicom
import cv2
import scipy.ndimage

CSV_PATH = "/workspace/alz mri_pet/metadata/final_dataset_expanded.csv"
TARGET_SHAPE = (32, 128, 128)

def load_dicom_series(folder):
    slices = []
    for f in os.listdir(folder):
        if f.endswith(".dcm"):
            path = os.path.join(folder, f)
            ds = pydicom.dcmread(path)
            slices.append(ds)

    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]) if "ImagePositionPatient" in x else 0)

    images = [s.pixel_array.astype(np.float32) for s in slices]
    volume = np.stack(images, axis=0)

    return volume

def normalize(volume):
    return (volume - np.mean(volume)) / (np.std(volume) + 1e-5)

def resize_volume(volume):
    factors = (
        TARGET_SHAPE[0] / volume.shape[0],
        TARGET_SHAPE[1] / volume.shape[1],
        TARGET_SHAPE[2] / volume.shape[2],
    )
    return scipy.ndimage.zoom(volume, factors, order=1)

def load_data():
    df = pd.read_csv(CSV_PATH)

    label_map = {"CN": 0, "MCI": 1, "AD": 2}

    data = []
    labels = []

    for _, row in df.iterrows():
        try:
            mri = load_dicom_series(row["mri_path"])
            pet = load_dicom_series(row["pet_path"])

            mri = resize_volume(mri)
            pet = resize_volume(pet)

            mri = normalize(mri)
            pet = normalize(pet)

            sample = np.stack([mri, pet], axis=0)

            data.append(sample)
            labels.append(label_map[row["label"]])

        except Exception as e:
            print("skip:", e)
            continue

    return np.array(data), np.array(labels)

if __name__ == "__main__":
    data, labels = load_data()

    print("Data shape:", data.shape)
    print("Labels:", np.bincount(labels))