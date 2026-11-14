"""STL-10 v2: ResNet-18 transfer learning with the input resolution, augmentation and
schedule it needs.

What changes compared with the original notebook:
  * images are resized to IMG_SIZE (default 160) instead of fed at 96x96 -- ResNet-18 was
    pretrained on 224x224 ImageNet crops, and at 96 px its final feature map is only 3x3;
  * training images are augmented (random resized crop, flip, light colour jitter) --
    the original fine-tune reached 99.95% train vs 85.5% val accuracy after 5 epochs;
  * all layers can be fine-tuned with a lower learning rate for the backbone than for
    the new head ("discriminative" learning rates), AdamW, warmup + cosine schedule,
    label smoothing, mixed precision;
  * the checkpoint with the best *validation* accuracy is kept, and the test set is
    evaluated exactly once at the end.

    python improvements/stl10_v2.py                       # full fine-tune, 15 epochs (~15 min on a T4)
    python improvements/stl10_v2.py --mode layer4         # same recipe, only layer4 + head trainable
    python improvements/stl10_v2.py --mode frozen         # linear probe on frozen features
    python improvements/stl10_v2.py --fake-data --epochs 1 --img-size 64   # smoke test
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import fit, get_device, save_results, seed_everything, test  # noqa: E402

CLASSES = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def make_datasets(args):
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(), normalize,
    ])
    eval_tf = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(), normalize])

    if args.fake_data:
        make = lambda n, tf: datasets.FakeData(n, (3, 96, 96), 10, transform=tf)  # noqa: E731
        return make(256, train_tf), make(128, eval_tf), make(128, eval_tf)

    train_aug = datasets.STL10(args.data, split="train", download=True, transform=train_tf)
    train_eval = datasets.STL10(args.data, split="train", download=False, transform=eval_tf)
    test_set = datasets.STL10(args.data, split="test", download=True, transform=eval_tf)
    # Same 4000/1000 split as the original notebook (random_split with seed 42).
    perm = torch.randperm(len(train_aug), generator=torch.Generator().manual_seed(42)).tolist()
    return Subset(train_aug, perm[:4000]), Subset(train_eval, perm[4000:]), test_set


def build_model(mode: str, pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    for name, p in model.named_parameters():
        if mode == "frozen":
            p.requires_grad = name.startswith("fc.")
        elif mode == "layer4":
            p.requires_grad = name.startswith(("fc.", "layer4."))
        else:  # full
            p.requires_grad = True
    return model


def warmup_cosine(optimizer, total_steps: int, warmup_steps: int):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["full", "layer4", "frozen"], default="full")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--fake-data", action="store_true", help="random images, no download, no pretrained weights")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or f"results/stl10_v2_{args.mode}.json"

    seed_everything(args.seed)
    device = get_device()
    train_set, val_set, test_set = make_datasets(args)
    loader = lambda ds, shuffle: DataLoader(ds, args.batch_size, shuffle=shuffle,  # noqa: E731
                                            num_workers=2, pin_memory=device.type == "cuda")
    train_loader, val_loader, test_loader = loader(train_set, True), loader(val_set, False), loader(test_set, False)

    model = build_model(args.mode, pretrained=not args.fake_data).to(device)
    head = [p for n, p in model.named_parameters() if n.startswith("fc.")]
    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.") and p.requires_grad]
    groups = [{"params": head, "lr": args.head_lr}]
    if backbone:
        groups.append({"params": backbone, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    total = args.epochs * len(train_loader)
    scheduler = warmup_cosine(optimizer, total, warmup_steps=max(1, len(train_loader)))

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"mode={args.mode} | img={args.img_size}px | trainable params={n_train:,} | device={device}")

    history, best_epoch = fit(model, train_loader, val_loader, optimizer, scheduler, args.epochs, device,
                              label_smoothing=args.label_smoothing)
    test_loss, test_acc, preds, targets = test(model, test_loader, device)
    print(f"\nbest val epoch {best_epoch} | test loss {test_loss:.4f} | test accuracy {test_acc:.4f}\n")
    labels = list(range(len(CLASSES)))
    print(classification_report(targets, preds, labels=labels, target_names=CLASSES, digits=4, zero_division=0))

    save_results(args.out, dict(config=vars(args), best_epoch=best_epoch, test_loss=test_loss, test_acc=test_acc,
                                confusion_matrix=confusion_matrix(targets, preds, labels=labels).tolist(),
                                history=history))


if __name__ == "__main__":
    main()
