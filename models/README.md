# Overview
This folder contains PyTorch neural network models for 1D signal tasks such as classification and unsupervised learning. The models demonstrate key machine learning (ML) and deep learning (DL) concepts, including convolutional neural networks (CNNs), autoencoders, and structured training pipelines.

---

## Core Concepts

1. Supervised Learning – Classification
   The `CNN1D_Small` and `CNN1D_Large` models classify 1D input signals (e.g., time-domain pulses) into one of several classes using supervised learning. Training requires labeled data.

2. Unsupervised Learning – Autoencoding
   Autoencoders (`Autoencoder`, `CNNAutoencoder`) learn to compress and reconstruct input data without labels. These models are useful for dimensionality reduction, denoising, and anomaly detection.

---

## Neural Network Architectures

### CNN Models
   - Use `Conv1d` layers to extract local temporal features.
   - Max pooling and global average pooling reduce data dimensionality.
   - `GroupNorm` normalizes groups of channels, improving training with small batch sizes.
   - Dropout helps prevent overfitting.
   - `CNN1D_Large` has greater depth and normalization than `CNN1D_Small`.

### Fully Connected Autoencoder
   - `FCEncoder` and `FCDecoder` use stacked `Linear` layers with `ReLU` activations.
   - Learns dense latent representations of input signals.

### CNN Autoencoder
   - `CNNEncoder` uses stacked `Conv1d` layers to reduce dimensionality.
   - `CNNDecoder` uses `ConvTranspose1d` layers to reconstruct the signal.
   - Latent space is a compressed encoding of the original signal.

---

## Training Pipeline

- All models are trained using standard PyTorch loops (`train_model`, `train_autoencoder`).
- Training supports both CPU and GPU execution (`model.to(device)`).
- Optimizer: Adam with optional learning rate schedulers.
- Classification training tracks accuracy; autoencoder training tracks MSE reconstruction loss.

---

## Metrics

- Accuracy for classification models.
- MSELoss for autoencoders.
- Optional validation during training helps monitor generalization.
- Learning rate recorded per epoch for scheduler tracking.

---

## Model Summary

| Model            | Type             | Description                        |
|------------------|------------------|------------------------------------|
| CNN1D_Small      | Classifier (CNN) | Simple 3-layer CNN with dropout    |
| CNN1D_Large      | Classifier (CNN) | Deeper CNN with GroupNorm          |
| Autoencoder      | FC Autoencoder   | Fully connected encoder-decoder    |
| CNNAutoencoder   | CNN Autoencoder  | CNN-based signal reconstruction    |

---

## References

- PyTorch: https://pytorch.org
- GroupNorm paper: https://arxiv.org/abs/1803.08494
- Deep Learning by Goodfellow et al.

---

## Notes

- Input data should be shaped as (batch_size, 1, 1024).
- GroupNorm is used over BatchNorm for small-batch training stability.
- Code is modular and easy to extend with additional architectures or tasks.