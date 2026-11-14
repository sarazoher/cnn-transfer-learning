# Experiment log

All numbers are from single training runs, as recorded during the course assignments.

## CIFAR-10: CNN from scratch (12 ablations)

40,000 / 10,000 train/validation split of the CIFAR-10 training set; the official 10,000-image test
set. Each experiment changes one or two things relative to the best model so far.

| # | Experiment | Conv layers | Dense (dropout) | BN | Act. | Init | Optimiser (LR) | Train acc | Val acc | Test acc | Test F1 | Conclusion |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---|
| 1 | Small Baseline CNN | 32-64 | 64 (0.3) | no | ReLU | He | SGD+Nesterov (0.01) | 83.0% | 70.8% | 70.6% | 0.705 | First working CNN baseline, too weak; visible overfitting, good as starting point |
| 2 | Conv2D(4) CNN | 32-32-64-64 | 256 (0.3) | no | ReLU | He | SGD+Nesterov (0.01) | 89.5% | 73.9% | 73.3% | 0.725 | Better than E1, still overfitting, new temporary baseline. |
| 3 | (4)Conv2D CNN + BatchNorm | 32-32-64-64 | 256 (0.3) | yes | ReLU | He | SGD+Nesterov (0.01) | 90.5% | 79.1% | 78.3% | 0.782 | BatchNorm helped a lot, best one so far |
| 4 | Conv2D CNN + BatchNorm + Adam | 32-32-64-64 | 256 (0.3) | yes | ReLU | He | Adam (0.001) | 82.5% | 79.7% | 79.0% | 0.790 | Adam beat SGD on this architecture, cleaner optimizer |
| 5 | Stronger 6Conv CNN + BN + 2Dense + Adam | 32-32-64-64-128-128 | 256 (0.3), 128 (0.4) | yes | ReLU | He | Adam (0.001) | 91.3% | 83.0% | 82.0% | 0.818 | Current best model so far, new main architecture |
| 6 | Stronger 6Conv CNN + BN + 2Dense + SGD + Momentum + Nesterov | 32-32-64-64-128-128 | 256 (0.3), 128 (0.4) | yes | ReLU | He | SGD+Nesterov (0.01) | 93.0% | 82.9% | 80.5% | 0.804 | Adam still better than SGD also on this model |
| 7 | Strong model + Stronger Dropout | 32-32-64-64-128-128 | 256 (0.4), 128 (0.5) | yes | ReLU | He | Adam (0.001) | 89.3% | 78.1% | 76.0% | 0.763 | Stronger dropout hurt, preferred dropout remains 0.3 / 0.4 |
| 8 | Xavier Initialization + Stronger Dropout | 32-32-64-64-128-128 | 256 (0.4), 128 (0.5) | yes | ReLU | Xavier | Adam (0.001) | 92.7% | 80.1% | 79.7% | 0.798 | Combined changed experiment, dropout and Xavier |
| 9 | Strong model + Xavier | 32-32-64-64-128-128 | 256 (0.3), 128 (0.4) | yes | ReLU | Xavier | Adam (0.001) | 91.6% | 77.1% | 76.5% | 0.768 | He/Kaiming clearly better than Xavier on the best architecture |
| 10 | SGD + Momentum + Nesterov with LeakyReLU Activation | 32-32-64-64-128-128 | 256 (0.3), 128 (0.4) | yes | LeakyReLU | He | SGD+Nesterov (0.001) | 93.8% | 77.4% | 76.1% | 0.763 | LeakyReLU + SGD underperformed Adam baseline. |
| 11 | LeakyReLU Activation | 32-32-64-64-128-128 | 256 (0.3), 128 (0.4) | yes | LeakyReLU | He | Adam (0.001) | 91.6% | 83.1% | **82.9%** | 0.829 | Very close to Experiment 5; slightly better test accuracy, slightly worse validation accuracy |
| 12 | Wider Dense Layer | 32-32-64-64-128-128 | 512 (0.3), 256 (0.4) | yes | LeakyReLU | He | Adam (0.001) | 91.4% | 82.7% | 80.6% | 0.807 | Wider dense hurt the performance, original size remains best |

### What the log shows

- **Clear wins:** going from 2 to 4 to 6 conv layers (70.6% → 73.3% → 82.0% test), adding batch
  normalisation (+5 points over the same 4-layer network), and He initialisation over Xavier
  (82.0% vs 76.5% on the same network).
- **Things that hurt:** stronger dropout (0.4/0.5) and wider dense layers.
- **Differences that are probably noise.** Experiments 5, 11 and 12 are within about 1–2 points of
  each other, and so are 5 and 6 on validation accuracy (83.0% vs 82.9%). A rerun of the
  Experiment 11 architecture and optimiser in the final notebook reached 80.6% test accuracy
  rather than 82.9%, so single-run differences of this size are not conclusive. Experiment 8
  (Xavier + stronger dropout) also beat Experiment 9 (Xavier alone), although Experiment 7 had
  found stronger dropout harmful, another sign of run-to-run variance.
- **Not tried:** data augmentation and learning-rate schedules, the two changes that usually
  matter most on CIFAR-10. Both are in experiment 13 below.

### Experiment 13 (v2, to run)

`improvements/cifar10_v2.py` keeps the Experiment 11 architecture and adds random crop + horizontal
flip, AdamW with a one-cycle learning-rate schedule, and best-validation checkpointing, and repeats
the run over 3 seeds to report mean ± standard deviation. Run it with `--no-augment` as well to
measure augmentation on its own.

## STL-10: transfer learning vs. training from scratch

5,000 labelled training images split 4,000 / 1,000 (seed 42), 8,000 test images, 100,000 unlabelled
images for the autoencoder.

| Experiment | Model | Pretraining | Trainable | Optimiser (LR) | Epochs | Train acc | Val acc | Test acc | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---|
| Custom CNN from scratch | Custom CNN | None | All layers trainable | Adam (0.001) | 5 | 48.8% | 48.8% | 51.2% | Trained from scratch, short training time, validation and test accuracy around 50% |
| ResNet18 frozen feature extractor | ResNet18 | ImageNet | All pretrained layers frozen, only final FC trainable | Adam (0.001) | 5 | 81.9% | 78.9% | 78.2% | Frozen feature extractor, only classifier trained, much better validation accuracy than the CNN |
| ResNet18 partial fine-tuning | ResNet18 | ImageNet | Earlier layers frozen, layer4 and FC unfrozen | Adam (0.0001) | 5 | 99.95% | 85.5% | 86.3% | Partial fine-tuning, lower LR because pretrained weights are updated |
| Autoencoder pretraining only | Conv autoencoder | STL-10 unlabeled images | All autoencoder layers trainable | Adam (0.001) | 2 | – | – | – | Trained autoencoder on 100,000 unlabeled STL-10 images for 2 epochs. This was reconstruction only, not classification. Final reconstruction loss = 0.009601. |
| Autoencoder encoder classifier | Conv autoencoder encoder + classifier | STL-10 unlabeled images via autoencoder | Encoder frozen, classifier head trained | Adam (0.001) | 5 | 25.6% | 32.8% | 32.9% | The encoder was pretrained with an autoencoder for 2 epochs on the unlabeled STL-10 images. Then the encoder was frozen and only the classifier was trained for 5 epochs. The accuracy was low, probably because the autoencoder was trained for a short time and reconstruction learning does not directly learn class differences. |

### v2 (to run)

`improvements/stl10_v2.py`: images resized to 160 px, random-resized-crop / flip / colour-jitter
augmentation, full fine-tuning with a 10x lower learning rate for the backbone than the head,
warmup + cosine schedule, label smoothing, 15 epochs, best-validation checkpointing. `--mode layer4`
and `--mode frozen` rerun the original two transfer-learning set-ups with the same recipe, so the
effect of the recipe and of the number of fine-tuned layers can be separated.
