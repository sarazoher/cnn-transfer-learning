# CNNs from scratch vs. transfer learning (CIFAR-10, STL-10)

Two image-classification studies in PyTorch, started as Deep Learning course projects:

1. **CIFAR-10, CNN from scratch:** a 12-experiment ablation over depth, batch
   normalisation, optimiser, dropout, initialisation, activation and layer width, ending at
   **82.9% test accuracy** with a 6-layer CNN.
2. **STL-10, little labelled data:** a CNN from scratch, ImageNet-pretrained ResNet-18
   (frozen and partially fine-tuned), and self-supervised pretraining with a convolutional
   autoencoder on 100k unlabelled images. Partial fine-tuning reached **86.3% test accuracy**
   from 4,000 labelled images.

`improvements/` holds a second iteration that addresses the weaknesses found in the first
(no augmentation, single runs, wrong input resolution for ResNet).

## Results

### CIFAR-10: what each change did (test accuracy)

| Step | Test acc |
|---|---:|
| 2 conv layers, SGD + Nesterov | 70.6% |
| 4 conv layers | 73.3% |
| + batch normalisation | 78.3% |
| + Adam instead of SGD | 79.0% |
| 6 conv layers + 2 dense layers | 82.0% |
| He init → Xavier init (reverted) | 76.5% |
| stronger dropout 0.4/0.5 (reverted) | 76.0% |
| ReLU → LeakyReLU | **82.9%** |
| wider dense layers 512/256 (reverted) | 80.6% |

All 12 experiments, with train/validation accuracy, F1 and notes, are in
[EXPERIMENTS.md](EXPERIMENTS.md). That file also discusses which of the smaller
differences are within run-to-run noise.

### STL-10: 4,000 labelled training images

| Approach | Trainable params | Test acc |
|---|---:|---:|
| Custom CNN from scratch | 2.75M | 51.2% |
| ResNet-18 (ImageNet), frozen, linear head only | 5k | 78.2% |
| **ResNet-18 (ImageNet), `layer4` + head fine-tuned** | 8.4M | **86.3%** |
| Conv autoencoder on 100k unlabelled images → frozen encoder + classifier | 2.4M | 32.9% |

With this little labelled data, pretrained features dominate: a single linear layer on
frozen ImageNet features beats a CNN trained from scratch by 27 points.

## What I would do differently, and did in v2

| Weakness in v1 | Fix in `improvements/` |
|---|---|
| No data augmentation in either study | Random crop + flip (CIFAR-10); random resized crop + flip + colour jitter (STL-10) |
| Fine-tuned ResNet hit 99.95% train vs. 85.5% val accuracy: overfitting | Augmentation, label smoothing, weight decay, best-validation checkpointing |
| STL-10 images fed to ResNet-18 at 96 px (pretrained at 224 px; final feature map only 3×3) | Resize to 160 px (configurable) |
| Constant learning rate | One-cycle (CIFAR-10), warmup + cosine (STL-10) |
| Single runs, so 1–2 point differences are hard to interpret | `cifar10_v2.py` repeats over seeds and reports mean ± std |
| Autoencoder trained 2 epochs, encoder frozen | Discussed in the STL-10 notebook; the next step would be fine-tuning the encoder |

## Repository layout

```
notebooks/
  cifar10_cnn_ablations.ipynb     final CIFAR-10 model (Colab, executed)
  stl10_transfer_learning.ipynb   all four STL-10 approaches (Colab, executed)
improvements/
  common.py                       training loop: AMP, schedulers, best-val checkpointing
  cifar10_v2.py                   experiment 13: augmentation + one-cycle LR, multiple seeds
  stl10_v2.py                     ResNet-18 v2 recipe; --mode full | layer4 | frozen
EXPERIMENTS.md                    full experiment log and analysis
```

## Running

The notebooks are written for Google Colab with a GPU. For the v2 scripts:

```bash
pip install -r requirements.txt
python improvements/cifar10_v2.py              # 3 seeds x 40 epochs
python improvements/stl10_v2.py --mode full    # 15 epochs at 160 px
python improvements/stl10_v2.py --fake-data --epochs 1 --img-size 64   # quick check, no downloads
```

In Colab: `!git clone <this repo> && cd <repo> && python improvements/stl10_v2.py`.
Results are written to `results/*.json`.
