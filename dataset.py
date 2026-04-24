import os
import pandas as pd
from pathlib import Path

BASE = Path("/workspace/alz mri_pet/dataset/extracted/ADNI")
CSV_PATH = "/workspace/alz mri_pet/fffffffff_4_01_2026.csv"
OUT = "/workspace/alz mri_pet/metadata/final_dataset.csv"

df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

label_map = dict(zip(df["Subject"].astype(str), df["Group"]))

rows = []

for subject in os.listdir(BASE):
    subject_path = BASE / subject

    if subject not in label_map:
        continue

    label = label_map[subject]

    mri_scans = []
    pet_scans = []

    for folder in os.listdir(subject_path):
        if "MPRAGE" in folder.upper() or "MP-RAGE" in folder.upper():
            folder_path = subject_path / folder

            for date in os.listdir(folder_path):
                date_path = folder_path / date

                for img_id in os.listdir(date_path):
                    final_path = date_path / img_id

                    if any(f.endswith(".dcm") for f in os.listdir(final_path)):
                        mri_scans.append((date, str(final_path)))

    for folder in os.listdir(subject_path):
        if "PET" in folder.upper():
            folder_path = subject_path / folder

            for date in os.listdir(folder_path):
                date_path = folder_path / date

                for img_id in os.listdir(date_path):
                    final_path = date_path / img_id

                    if any(f.endswith(".dcm") for f in os.listdir(final_path)):
                        pet_scans.append((date, str(final_path)))

    if not mri_scans or not pet_scans:
        continue

    best = None
    best_diff = None

    for m_date, m_path in mri_scans:
        for p_date, p_path in pet_scans:
            try:
                d1 = pd.to_datetime(m_date.split("_")[0])
                d2 = pd.to_datetime(p_date.split("_")[0])
                diff = abs((d1 - d2).days)
            except:
                continue
    
            if best is None or diff < best_diff:
                best = (m_path, p_path, diff)
                best_diff = diff

    rows.append({
        "subject_id": subject,
        "label": label,
        "mri_path": best[0],
        "pet_path": best[1],
        "date_diff": best[2]
    })

final_df = pd.DataFrame(rows)
final_df.to_csv(OUT, index=False)

print("Final subjects:", len(final_df))
print(final_df.head())