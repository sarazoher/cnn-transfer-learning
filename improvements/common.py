"""Shared training utilities for the v2 experiments (CIFAR-10 and STL-10)."""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, scaler=None):
    """One pass over ``loader``. Trains if an optimizer is given, otherwise evaluates.

    Returns (mean loss, accuracy, predictions, targets).
    """
    training = optimizer is not None
    model.train(training)
    use_amp = device.type == "cuda"
    total_loss, correct, seen = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()  # per-batch schedulers (OneCycle / warmup-cosine)
            preds = logits.argmax(1)
            total_loss += loss.item() * images.size(0)
            correct += (preds == targets).sum().item()
            seen += images.size(0)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    return total_loss / seen, correct / seen, torch.cat(all_preds), torch.cat(all_targets)


def fit(model, train_loader: DataLoader, val_loader: DataLoader, optimizer, scheduler,
        epochs: int, device, label_smoothing: float = 0.0, patience: int | None = None):
    """Train with mixed precision, keep the weights with the best validation accuracy.

    The test set is never touched here: model selection uses validation only.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best_acc, best_state, best_epoch, history = -1.0, None, 0, []

    for epoch in range(1, epochs + 1):
        start = time.time()
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, device, optimizer, scheduler, scaler)
        va_loss, va_acc, _, _ = run_epoch(model, val_loader, eval_criterion, device)
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss, val_acc=va_acc))
        flag = ""
        if va_acc > best_acc:
            best_acc, best_epoch, flag = va_acc, epoch, "  *best*"
            best_state = copy.deepcopy(model.state_dict())
        print(f"epoch {epoch:>3}/{epochs} | train {tr_loss:.4f} / {tr_acc:.4f} | "
              f"val {va_loss:.4f} / {va_acc:.4f} | {time.time() - start:5.1f}s{flag}", flush=True)
        if patience and epoch - best_epoch >= patience:
            print(f"early stopping: no val improvement for {patience} epochs")
            break

    model.load_state_dict(best_state)
    return history, best_epoch


def test(model, loader, device):
    loss, acc, preds, targets = run_epoch(model, loader, nn.CrossEntropyLoss(), device)
    return loss, acc, preds.numpy(), targets.numpy()


def save_results(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))
    print(f"results written to {path}")
