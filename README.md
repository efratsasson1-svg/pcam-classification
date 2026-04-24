# PCam Histopathology Classification

This project focuses on binary classification of histopathology images from the **PatchCamelyon (PCam)** dataset using deep learning models.

## Dataset

- **PatchCamelyon (PCam)**
- Link: https://patchcamelyon.grand-challenge.org/
- Task: classify image patches as **tumor / non-tumor**

---

## Models

We compare three approaches:

### 1. ResNet (CNN baseline)
- Input size: 96×96
- Strong performance on local features

### 2. Phikon-v2 (Foundation Model)
- Pretrained pathology model
- Tested with frozen backbone + classifier
- Limited performance in this setup

### 3. DINO ViT-S (Vision Transformer)
- Fully fine-tuned
- Input resized to 224×224
- Best performing model

---

## Results

| Model | Test AUC |
|------|---------|
| ResNet18 | ~0.93 |
| Phikon-v2 | ~0.72 |
| DINO ViT-S | **~0.94–0.95** |

---

##  Project Structure
pcam_foundation_train.py - training pipeline  
PyTorch_Dataset.py - dataset loader  
print_Confusion_Matrix.py - evaluation utilities  
README.md  

## How to Run

1. Install dependencies:

```bash
pip install torch torchvision timm scikit-learn matplotlib seaborn
```
Prepare dataset (PCam)
Run training:
```bash
python pcam_foundation_train.py
```

---

## Evaluation

AUC (primary metric)
Accuracy
Confusion matrix visualization

---

## Notes

Dataset is not included in the repository
Outputs and checkpoints are excluded via .gitignore

---
## Author

Efrat Sasson
