"""CIFAR-10, experiment 13: the best architecture from the ablation log (Exp. 11) plus
data augmentation and a one-cycle learning-rate schedule, repeated over several seeds.

Why several seeds: experiments 5, 11 and 12 differ by 1-2 points of test accuracy, which is
about the size of run-to-run noise. Reporting mean +/- std over seeds shows which
differences are real.

    python improvements/cifar10_v2.py                        # 3 seeds x 40 epochs (~25 min on a T4)
    python improvements/cifar10_v2.py --seeds 0 --epochs 30
    python improvements/cifar10_v2.py --no-augment           # same schedule, no augmentation
    python improvements/cifar10_v2.py --fake-data --epochs 1 # smoke test, no download
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import fit, get_device, save_results, seed_everything, test  # noqa: E402

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


class SimpleCNN(nn.Module):
    """Experiment 11 architecture: 6 conv (32-32-64-64-128-128) + BN + LeakyReLU, 2 dense layers."""

    def __init__(self, num_classes: int = 10, dropout1: float = 0.3, dropout2: float = 0.4):
        super().__init__()

        def block(c_in, c_out):
            return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.LeakyReLU(0.01)]

        self.features = nn.Sequential(
            *block(3, 32), *block(32, 32), nn.MaxPool2d(2),
            *block(32, 64), *block(64, 64), nn.MaxPool2d(2),
            *block(64, 128), *block(128, 128), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.LeakyReLU(0.01), nn.Dropout(dropout1),
            nn.Linear(256, 128), nn.LeakyReLU(0.01), nn.Dropout(dropout2),
            nn.Linear(128, num_classes),
        )
        for m in self.modules():  # He initialisation, as in the ablations
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu", a=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))


def make_datasets(args):
    normalize = transforms.Normalize(MEAN, STD)
    train_tf = transforms.Compose(
        ([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if args.augment else [])
        + [transforms.ToTensor(), normalize])
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])

    if args.fake_data:
        make = lambda n, tf: datasets.FakeData(n, (3, 32, 32), 10, transform=tf)  # noqa: E731
        return make(512, train_tf), make(512, eval_tf), make(256, eval_tf)

    train_aug = datasets.CIFAR10(args.data, train=True, download=True, transform=train_tf)
    train_eval = datasets.CIFAR10(args.data, train=True, download=False, transform=eval_tf)
    test_set = datasets.CIFAR10(args.data, train=False, download=True, transform=eval_tf)
    # Same 40k/10k split for every seed; validation images are never augmented.
    perm = torch.randperm(len(train_aug), generator=torch.Generator().manual_seed(42)).tolist()
    n_val = int(len(perm) * args.val_fraction)
    return Subset(train_aug, perm[n_val:]), Subset(train_eval, perm[:n_val]), test_set


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--fake-data", action="store_true", help="random images, for a quick smoke test")
    ap.add_argument("--out", default="results/cifar10_v2.json")
    args = ap.parse_args()

    device = get_device()
    train_set, val_set, test_set = make_datasets(args)
    loader = lambda ds, shuffle: DataLoader(ds, args.batch_size, shuffle=shuffle,  # noqa: E731
                                            num_workers=2, pin_memory=device.type == "cuda")
    runs = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} | augmentation={'on' if args.augment else 'off'} | device={device} ===")
        seed_everything(seed)
        train_loader, val_loader, test_loader = loader(train_set, True), loader(val_set, False), loader(test_set, False)
        model = SimpleCNN().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.max_lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
        history, best_epoch = fit(model, train_loader, val_loader, optimizer, scheduler, args.epochs, device)
        test_loss, test_acc, preds, targets = test(model, test_loader, device)
        f1 = f1_score(targets, preds, average="macro")
        print(f"seed {seed}: best val epoch {best_epoch} | test acc {test_acc:.4f} | macro F1 {f1:.4f}")
        runs.append(dict(seed=seed, best_epoch=best_epoch, test_acc=test_acc, test_f1=f1,
                         best_val_acc=max(h["val_acc"] for h in history), history=history))

    accs = np.array([r["test_acc"] for r in runs])
    f1s = np.array([r["test_f1"] for r in runs])
    print(f"\nTest accuracy over {len(runs)} seed(s): {accs.mean():.4f} +/- {accs.std():.4f}")
    print(f"Test macro F1 over {len(runs)} seed(s): {f1s.mean():.4f} +/- {f1s.std():.4f}")
    save_results(args.out, dict(config=vars(args), test_acc_mean=accs.mean(), test_acc_std=accs.std(),
                                test_f1_mean=f1s.mean(), test_f1_std=f1s.std(), runs=runs))


if __name__ == "__main__":
    main()
