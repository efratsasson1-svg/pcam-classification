# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 20:39:39 2026

@author: efrat.sasson
"""

import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(model, loader, device, class_names=None):
    model.eval()
    model.to(device)

    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy().ravel())
            y_pred.extend(preds.cpu().numpy().ravel())

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names is not None else "auto",
        yticklabels=class_names if class_names is not None else "auto",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return cm
