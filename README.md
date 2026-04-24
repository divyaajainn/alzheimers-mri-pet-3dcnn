# Multimodal Alzheimer’s Disease Classification (MRI + PET)

This project implements a 3D CNN-based multimodal framework for Alzheimer’s Disease classification using MRI and PET scans.

## Approach
- MRI and PET processed separately
- Late Fusion architecture
- 3D CNN for volumetric feature extraction

## Dataset
ADNI dataset (not included due to size constraints)

- 73 subjects
- 90 MRI-PET pairs

## Models Included
- Baseline 3D CNN
- Attention 3D CNN (experiment)
- GAN (experiment)
- Final Late Fusion Model

## Result
Final model achieves ~72% validation accuracy.

## Note
Update dataset paths before running the code.
