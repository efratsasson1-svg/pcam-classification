# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:19:09 2026

@author: efrat.sasson
"""

#!/usr/bin/env python3
"""
pcam_foundation_train.py

Train a pathology foundation model on PatchCamelyon (PCam) from H5 files.

Stages:
  1) Frozen backbone + linear head
  2) Frozen backbone + MLP head
  3) Partially unfreeze backbone + MLP head

Example:
  python pcam_foundation_train.py \
      --data_dir /path/to/pcam \
      --model_name owkin/phikon-v2 \
      --batch_size 32 \
      --epochs_stage1 5 \
      --epochs_stage2 5 \
      --epochs_stage3 3 \
      --num_workers 0 \
      --output_dir ./outputs

Notes:
- Start with num_workers=0 for H5 stability.
- Uses ROC-AUC as primary metric.
- By default, each stage starts from the pretrained backbone again for fair comparison.
"""
import os

data_dir = "C:/Users/efrat.sasson/Downloads/pcamv1-20260405T105514Z-1-001/pcamv1/files"

for fname in [
    "camelyonpatch_level_2_split_train_x.h5",
    "camelyonpatch_level_2_split_train_y.h5",
    "camelyonpatch_level_2_split_valid_x.h5",
    "camelyonpatch_level_2_split_valid_y.h5",
    "camelyonpatch_level_2_split_test_x.h5",
    "camelyonpatch_level_2_split_test_y.h5",
]:
    full = os.path.join(data_dir, fname)
    print(fname, "->", os.path.exists(full), full)
import copy
import math
import json
import time
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List

import h5py
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import AutoImageProcessor, AutoModel

from sklearn.metrics import roc_auc_score

from torchvision.models import resnet18, ResNet18_Weights

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torchvision.models import resnet18

import timm
import torch.nn as nn

# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

# ----------------------------
# Config
# ----------------------------

@dataclass
class TrainConfig:
    data_dir: str = "C:/Users/efrat.sasson/Downloads/pcamv1-20260405T105514Z-1-001/pcamv1/files"
    model_name: str = "owkin/phikon-v2"
    output_dir: str = "./outputs"
    batch_size: int = 64
    num_workers: int = 0
    seed: int = 42
    image_size: int = 224
    lr_stage1: float = 3e-5
    lr_stage2: float = 1e-3
    lr_stage3_head: float = 1e-4
    lr_stage3_backbone: float = 1e-5
    weight_decay: float = 1e-4
    epochs_stage1: int = 5
    epochs_stage2: int = 5
    epochs_stage3: int = 3
    early_stopping_patience: int = 3
    mlp_hidden_dim: int = 512
    dropout: float = 0.2
    train_subset: int = 0   # 0 = full dataset
    valid_subset: int = 0
    test_subset: int = 0
    amp: bool = True
    run_stages: str = "1"   # e.g. "1,2,3" or "1"
    unfreeze_last_n_blocks: int = 1
    model_type: str = "fm"   # "fm" or "resnet"
# ----------------------------
# Dataset
# ----------------------------

class PCamH5Dataset(Dataset):
    def __init__(
        self,
        x_path: str,
        y_path: str,
        transform=None,
        subset_size: int = 0,
    ):
        self.x_file = h5py.File(x_path, "r")
        self.y_file = h5py.File(y_path, "r")
        self.x = self.x_file["x"]
        self.y = self.y_file["y"]
        self.transform = transform

        self.indices = np.arange(len(self.x))
        if subset_size > 0:
            self.indices = self.indices[:subset_size]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        img = self.x[real_idx]        # HWC uint8
        label = int(self.y[real_idx][0])

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.tensor(img).permute(2, 0, 1).float() / 255.0

        return img, label


def build_transforms(image_size: int, model_type: str):
    if model_type == "resnet":
        # NO resize — keep 96x96
        train_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])

        eval_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
    if model_type == "dino":
        train_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        eval_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
            ])
    else:
        # Foundation model → resize to 224
        train_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])

        eval_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    return train_tf, eval_tf


def build_dataloaders(cfg: TrainConfig):
    train_tf, eval_tf = build_transforms(cfg.image_size, cfg.model_type)

    train_ds = PCamH5Dataset(
        x_path=os.path.join(cfg.data_dir, "camelyonpatch_level_2_split_train_x.h5"),
        y_path=os.path.join(cfg.data_dir, "camelyonpatch_level_2_split_train_y.h5"),
        transform=train_tf,
        subset_size=cfg.train_subset,
    )

    valid_ds = PCamH5Dataset(
        x_path=os.path.join(cfg.data_dir, "camelyonpatch_level_2_split_valid_x.h5"),
        y_path=os.path.join(cfg.data_dir, "camelyonpatch_level_2_split_valid_y.h5"),
        transform=eval_tf,
        subset_size=cfg.valid_subset,
    )

    test_ds = PCamH5Dataset(
        x_path=os.path.join(cfg.data_dir, "camelyonpatch_level_2_split_test_x.h5"),
        y_path=os.path.join(cfg.data_dir, "camelyonpatch_level_2_split_test_y.h5"),
        transform=eval_tf,
        subset_size=cfg.test_subset,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, valid_loader, test_loader


# ----------------------------
# Model components
# ----------------------------

class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class FMClassifier(nn.Module):
    def __init__(self, backbone: AutoModel, head: nn.Module, processor: AutoImageProcessor):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.processor = processor

    def forward(self, images: torch.Tensor):
        """
        images: [B, 3, H, W], float in [0,1]
        """
        # HF image processors often accept tensors directly in channels-first format.
        inputs = self.processor(images=images, return_tensors="pt")
        device = images.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.backbone(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feats = outputs.pooler_output
        else:
            feats = outputs.last_hidden_state[:, 0, :]  # CLS token

        logits = self.head(feats)
        return logits

class DinoClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch16_224.dino",
            pretrained=True,
            num_classes=0
        )
        self.head = nn.Linear(self.backbone.num_features, 2)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)
    
def infer_feature_dim(backbone: AutoModel, processor: AutoImageProcessor, image_size: int, device: torch.device) -> int:
    dummy = torch.rand(2, 3, image_size, image_size, device=device)
    with torch.no_grad():
        inputs = processor(images=dummy, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = backbone(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feats = outputs.pooler_output
        else:
            feats = outputs.last_hidden_state[:, 0, :]
    return feats.shape[1]


def freeze_all_backbone(backbone: nn.Module) -> None:
    for p in backbone.parameters():
        p.requires_grad = False


def unfreeze_last_n_blocks(backbone: nn.Module, n: int) -> None:
    """
    Generic best-effort unfreeze for common ViT structures.
    """
    freeze_all_backbone(backbone)

    # Common transformer block containers
    candidate_paths = [
        ("encoder.layer", lambda m: m.encoder.layer if hasattr(m, "encoder") and hasattr(m.encoder, "layer") else None),
        ("vit.encoder.layer", lambda m: m.vit.encoder.layer if hasattr(m, "vit") and hasattr(m.vit, "encoder") and hasattr(m.vit.encoder, "layer") else None),
        ("blocks", lambda m: m.blocks if hasattr(m, "blocks") else None),
    ]

    blocks = None
    for _, getter in candidate_paths:
        blocks = getter(backbone)
        if blocks is not None:
            break

    if blocks is None:
        # Fallback: unfreeze everything if we can't find blocks cleanly
        for p in backbone.parameters():
            p.requires_grad = True
        return

    if n <= 0:
        return

    for block in list(blocks)[-n:]:
        for p in block.parameters():
            p.requires_grad = True

    # Often helps to unfreeze final norm / layernorm if present
    for name, param in backbone.named_parameters():
        lname = name.lower()
        if "layernorm" in lname or lname.endswith("norm.weight") or lname.endswith("norm.bias"):
            param.requires_grad = True


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------
# Metrics / evaluation
# ----------------------------

def compute_metrics(y_true: List[int], y_prob: List[float], y_pred: List[int]) -> Dict[str, float]:
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    return {"acc": acc, "auc": auc}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true, y_prob, y_pred = [], [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * labels.size(0)
        y_true.extend(labels.cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    metrics = compute_metrics(y_true, y_prob, y_pred)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


# ----------------------------
# Training
# ----------------------------

def train_one_stage(
    stage_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    output_ckpt_path: str,
    use_amp: bool = True,
    patience: int = 3,
    scheduler=None,
    ) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and torch.cuda.is_available()))

    best_val_auc = -math.inf
    best_state = None
    best_epoch = -1
    epochs_without_improvement = 0

    history = []
    lr_history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        y_true, y_prob, y_pred = [], [], []
        
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(use_amp and torch.cuda.is_available())):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            probs = torch.softmax(logits, dim=1)[:, 1].detach()
            preds = torch.argmax(logits, dim=1).detach()

            running_loss += loss.item() * labels.size(0)
            y_true.extend(labels.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

        train_metrics = compute_metrics(y_true, y_prob, y_pred)
        train_metrics["loss"] = running_loss / len(train_loader.dataset)

        val_metrics = evaluate(model, valid_loader, device)
        
        log_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_auc": train_metrics["auc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_auc": val_metrics["auc"],
        }
        history.append(log_row)
        current_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(current_lr)
        if scheduler is not None:
            scheduler.step()
        print(
            f"[{stage_name}] epoch {epoch}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} train_auc={train_metrics['auc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} val_auc={val_metrics['auc']:.4f}"
        )
        print(f"Epoch {epoch+1} LR: {current_lr:.6f}")
        current_val_auc = val_metrics["auc"]
        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, output_ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[{stage_name}] early stopping at epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_metrics = evaluate(model, valid_loader, device)
    final_val_metrics["best_epoch"] = best_epoch

    history_path = output_ckpt_path.replace(".pt", "_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(lr_history)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid()
    plt.show()
    plt.plot(lr_history, marker='o')
    return final_val_metrics
    
# ----------------------------
# Stage builders
# ----------------------------

def build_stage_model(cfg: TrainConfig, stage: int, device: torch.device) -> Tuple[nn.Module, Dict[str, object]]:
    if cfg.model_type == "resnet":
        if stage != 1:
            raise ValueError("For model_type='resnet', use stage 1 only.")
        model = build_resnet_model().to(device)
        meta = {
            "feat_dim": None,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        return model, meta
    
    processor = AutoImageProcessor.from_pretrained(cfg.model_name)
    backbone = AutoModel.from_pretrained(cfg.model_name)

    feat_dim = infer_feature_dim(backbone, processor, cfg.image_size, device=torch.device("cpu"))

    if stage == 1:
        head = LinearHead(feat_dim, num_classes=2)
        freeze_all_backbone(backbone)

    elif stage == 2:
        head = MLPHead(
            feat_dim,
            hidden_dim=cfg.mlp_hidden_dim,
            dropout=cfg.dropout,
            num_classes=2,
        )
        freeze_all_backbone(backbone)

    elif stage == 3:
        head = MLPHead(
            feat_dim,
            hidden_dim=cfg.mlp_hidden_dim,
            dropout=cfg.dropout,
            num_classes=2,
        )
        unfreeze_last_n_blocks(backbone, cfg.unfreeze_last_n_blocks)

    else:
        raise ValueError(f"Unknown stage: {stage}")
    # ---- DINO ----
    if cfg.model_type == "dino":
        model = DinoClassifier().to(device)

        # make sure EVERYTHING is trainable
        for p in model.parameters():
           p.requires_grad = True

        meta = {
            "feat_dim": model.backbone.num_features,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            }

    return model, meta

    # ---- ResNet ----
    if cfg.model_type == "resnet":
        if stage != 1:
            raise ValueError("For model_type='resnet', use stage 1 only.")
        model = build_resnet_model().to(device)
        meta = {
            "feat_dim": None,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        return model, meta
    model = FMClassifier(backbone=backbone, head=head, processor=processor)
    model.to(device)

    meta = {
        "feat_dim": feat_dim,
        "trainable_params": count_trainable_params(model),
    }
    return model, meta

def build_optimizer(cfg: TrainConfig, model: nn.Module, stage: int):
    if stage in (1, 2):
        lr = cfg.lr_stage1 if stage == 1 else cfg.lr_stage2
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)

    if stage == 3:
        head_params = []
        backbone_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("head."):
                head_params.append(p)
            else:
                backbone_params.append(p)

        param_groups = []
        if head_params:
            param_groups.append({"params": head_params, "lr": cfg.lr_stage3_head})
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": cfg.lr_stage3_backbone})

        return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    if cfg.model_type == "dino":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr_stage1,
            weight_decay=cfg.weight_decay
            )
    raise ValueError(f"Unknown stage: {stage}")


# ----------------------------
# Main
# ----------------------------

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--data_dir",
    type=str,
    default="C:/Users/efrat.sasson/Downloads/pcamv1-20260405T105514Z-1-001/pcamv1/files"
    )
    parser.add_argument("--model_name", type=str, default="owkin/phikon-v2")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--lr_stage1", type=float, default=1e-3)
    parser.add_argument("--lr_stage2", type=float, default=1e-3)
    parser.add_argument("--lr_stage3_head", type=float, default=1e-4)
    parser.add_argument("--lr_stage3_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--epochs_stage1", type=int, default=5)
    parser.add_argument("--epochs_stage2", type=int, default=5)
    parser.add_argument("--epochs_stage3", type=int, default=3)
    parser.add_argument("--early_stopping_patience", type=int, default=3)

    parser.add_argument("--mlp_hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--train_subset", type=int, default=0)
    parser.add_argument("--valid_subset", type=int, default=0)
    parser.add_argument("--test_subset", type=int, default=0)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.set_defaults(amp=True)

    parser.add_argument("--run_stages", type=str, default="1,2,3")
    parser.add_argument("--unfreeze_last_n_blocks", type=int, default=1)

    args = parser.parse_args()
    return TrainConfig(**vars(args))

  
def verify_files(cfg: TrainConfig) -> None:
    required = [
        "camelyonpatch_level_2_split_train_x.h5",
        "camelyonpatch_level_2_split_train_y.h5",
        "camelyonpatch_level_2_split_valid_x.h5",
        "camelyonpatch_level_2_split_valid_y.h5",
        "camelyonpatch_level_2_split_test_x.h5",
        "camelyonpatch_level_2_split_test_y.h5",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(cfg.data_dir, f))]
    if missing:
        raise FileNotFoundError(f"Missing required files in {cfg.data_dir}: {missing}")

    for split in ["train", "valid", "test"]:
        x_path = os.path.join(cfg.data_dir, f"camelyonpatch_level_2_split_{split}_x.h5")
        y_path = os.path.join(cfg.data_dir, f"camelyonpatch_level_2_split_{split}_y.h5")
        with h5py.File(x_path, "r") as fx, h5py.File(y_path, "r") as fy:
            print(f"{split}: x={fx['x'].shape} {fx['x'].dtype}, y={fy['y'].shape} {fy['y'].dtype}")

def build_resnet_model():
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def plot_roc_curve(model, loader, device, title="ROC Curve"):
    model.eval()
    
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")  # random baseline
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def main(cfg=None):
    if cfg is None:
        cfg = parse_args()
    ensure_dir(cfg.output_dir)
    set_seed(cfg.seed)

    print(f"[{now_str()}] starting")
    print(json.dumps(asdict(cfg), indent=2))

    verify_files(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, valid_loader, test_loader = build_dataloaders(cfg)

    run_stages = [int(x.strip()) for x in cfg.run_stages.split(",") if x.strip()]
    all_results = {}
    print("model_type:", cfg.model_type)
    print("model_name:", cfg.model_name)
    for stage in run_stages:
        print("=" * 80)
        print(f"Running stage {stage}")

        model, meta = build_stage_model(cfg, stage=stage, device=device)
        optimizer = build_optimizer(cfg, model, stage=stage)
        epochs = {
               1: cfg.epochs_stage1,
               2: cfg.epochs_stage2,
               3: cfg.epochs_stage3,
               }[stage]

        print("Trainable parameters:")
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name)
        print(f"stage {stage} feature_dim: {meta['feat_dim']}")
        print(f"stage {stage} trainable_params: {meta['trainable_params']:,}")
       

        ckpt_path = os.path.join(cfg.output_dir, f"stage{stage}.pt")

        epochs = {
            1: cfg.epochs_stage1,
            2: cfg.epochs_stage2,
            3: cfg.epochs_stage3,
        }[stage]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)

        val_metrics = train_one_stage(
            stage_name=f"stage{stage}",
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device,
            epochs=epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            output_ckpt_path=ckpt_path,
            use_amp=cfg.amp,
            patience=cfg.early_stopping_patience,
        )

        test_metrics = evaluate(model, test_loader, device)

        result = {
            "val": val_metrics,
            "test": test_metrics,
            "checkpoint": ckpt_path,
        }
        all_results[f"stage{stage}"] = result

        print(f"[stage{stage}] best val: {val_metrics}")
        print(f"[stage{stage}] test: {test_metrics}")

    results_path = os.path.join(cfg.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("=" * 80)
    print("Final results:")
    print(json.dumps(all_results, indent=2))
    print(f"saved results to: {results_path}")

    plot_roc_curve(model, test_loader, device, title="Test ROC Curve")

    from print_Confusion_Matrix import plot_confusion_matrix
    plot_confusion_matrix(model, test_loader, device)
if __name__ == "__main__":
    cfg = TrainConfig(
        data_dir="C:/Users/efrat.sasson/Downloads/pcamv1-20260405T105514Z-1-001/pcamv1/files",
        model_type="dino",
        run_stages="1",
        batch_size=32,
        num_workers=0,  # safer for H5
        lr_stage1 = 3e-5
    )
    main(cfg)