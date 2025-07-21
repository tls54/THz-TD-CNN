# THz-TD-CNN: Deep Learning Framework for THz Time-Domain Spectroscopy

This project provides a suite of tools and models for analyzing Terahertz Time-Domain Spectroscopy (THz-TDS) data using deep learning techniques. The primary objective is to classify materials or infer structural properties from time-domain signals using CNNs and encoder-decoder architectures.

---

## Project Structure

- **Basic_CNN_ident.ipynb** — Demonstrates CNN-based material classification.
- **Encoder-approach.ipynb** — Encoder-decoder model for dimensionality reduction.
- **Model_testing.ipynb** — Evaluation workflows for different architectures.
- **Data_collection.ipynb** — Scripts for testing THz datasets.
- **Simulate.py** — THz signal simulation back end.
- **finetuning.ipynb** — Fine-tune models in `trained_models/` on new or augmented datasets.
- **Time_domain_UMAP_projection.ipynb** — Visualization of pulse features using UMAP.
- **Multi-processing.py** — Script to leverage multiprocessing for efficient dataset generation.
- **TODO.md** — Project planning notes and tasks.

---

## Models

Stored under `models/`:
- **Classifier_models.py** — CNN-based classifiers.
- **Encoder_decoder.py** — Autoencoder architectures.
- **encoder_decoder_utils.py** — Utilities for encoder/decoder training.
- **utils.py** — General-purpose utilities for model training and evaluation.

Pretrained weights:
- `trained_models/`
The architecture these are for will be specified in the name of the weight file.

---

## Benchmarks

Located in `data_gen_benchmark/`, this module compares multiprocessing and system performance across different to ensure data generation is correctly parallelized.

- **compare_pc_mac.ipynb** — System-dependent benchmark analysis.
- **Benchmark_results_mac/** and **Modelling_pc_benchmarks/** — Contain plots and CSVs comparing average time per worker, samples/sec, and scaling behavior.

---

## How to Use

### 1. Simulate Data
Use `Simulate.py` or `data_gen_benchmark/Simulate.py` to create synthetic THz pulses.

### 2. Train Classifier
Run `Basic_CNN_ident.ipynb` to train or test a CNN on your dataset.

### 3. Visualize Latent Space
Use `Time_domain_UMAP_projection.ipynb` after training an encoder to explore learned representations.

---

## Requirements

Typical setup includes:
- Python ≥ 3.8
- PyTorch
- NumPy, SciPy
- Matplotlib, seaborn
- scikit-learn
- UMAP-learn


---

## Status & To-do

See `TODO.md` for planned features and improvements.

---

## Acknowledgments

Developed as part of a THz TDS deep learning research pipeline. For academic use and extensions.

