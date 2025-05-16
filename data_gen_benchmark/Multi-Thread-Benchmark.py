import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from Simulate import simulate_parallel, simulate_reference
from time import perf_counter
from math import ceil
import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuration
L = 2**12
DOWNSAMPLE_FACTOR = 4
NUM_SAMPLES = 60_000
LAYER_LIMS = [1, 3]
N_RANGE = (1.1, 6.0)
K_RANGE = (-0.1, 0.01)
D_RANGE = (0.05e-3, 0.5e-3)
OUTPUT_CSV = "benchmark_results.csv"

def downsample_tensor(tensor, factor):
    return tensor[::factor]

def generate_samples():
    return np.random.randint(LAYER_LIMS[0], LAYER_LIMS[1] + 1, NUM_SAMPLES)

def generate_material_parameters(total_layers):
    n_values = np.random.uniform(*N_RANGE, total_layers)
    k_values = np.random.uniform(*K_RANGE, total_layers)
    D_values = np.random.uniform(*D_RANGE, total_layers)
    return n_values, k_values, D_values

def construct_material_samples(samples, n_values, k_values, D_values):
    material_samples = []
    layer_index = 0
    for num_layers in samples:
        sample = []
        for _ in range(num_layers):
            n_complex = n_values[layer_index] + k_values[layer_index] * 1j
            D = D_values[layer_index]
            sample.append((n_complex, D))
            layer_index += 1
        material_samples.append(sample)
    return material_samples

def process_batch(args):
    reference_pulse, material_samples_batch, indices, deltat = args
    batch_results = []
    for i, idx in enumerate(indices):
        pulse = simulate_parallel(reference_pulse, material_samples_batch[i], deltat, 0)[1].detach().cpu()[:L]
        downsampled_pulse = downsample_tensor(pulse, DOWNSAMPLE_FACTOR)
        batch_results.append((downsampled_pulse, material_samples_batch[i], len(material_samples_batch[i])))
    return batch_results

def run_benchmark(worker_count):
    print(f"\nRunning benchmark with {worker_count} workers...")
    samples = generate_samples()
    total_layers = sum(samples)

    n_values, k_values, D_values = generate_material_parameters(total_layers)
    material_samples = construct_material_samples(samples, n_values, k_values, D_values)

    deltat = 0.0194e-12
    reference_pulse = simulate_reference(L, deltat)

    batch_size = ceil(len(samples) / worker_count)
    sample_batches = [samples[i*batch_size : min((i+1)*batch_size, len(samples))] for i in range(worker_count)]
    material_batches = [material_samples[i*batch_size : min((i+1)*batch_size, len(material_samples))] for i in range(worker_count)]
    index_batches = [list(range(i*batch_size, min((i+1)*batch_size, len(samples)))) for i in range(worker_count)]

    args = [(reference_pulse, material_batches[i], index_batches[i], deltat) for i in range(worker_count)]

    start = perf_counter()
    with Pool(processes=worker_count) as pool:
        results = list(tqdm(pool.imap_unordered(process_batch, args), total=worker_count, ncols=80))
    end = perf_counter()

    total_time = end - start
    samples_per_sec = NUM_SAMPLES / total_time
    time_per_worker = total_time / worker_count

    return {
        "workers": worker_count,
        "total_time": total_time,
        "samples_per_second": samples_per_sec,
        "time_per_worker": time_per_worker
    }

def plot_results(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df["workers"], df["total_time"], marker='o')
    plt.xlabel("Number of Workers")
    plt.ylabel("Total Time (s)")
    plt.title("Total Processing Time vs Number of Workers")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(df["workers"], df["samples_per_second"], marker='o', color='green')
    plt.xlabel("Number of Workers")
    plt.ylabel("Samples per Second")
    plt.title("Samples per Second vs Number of Workers")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(df["workers"], df["time_per_worker"], marker='o', color='red')
    plt.xlabel("Number of Workers")
    plt.ylabel("Average Time per Worker (s)")
    plt.title("Average Time per Worker vs Number of Workers")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results = []
    MAX_WORKERS = 11
    for workers in range(1, MAX_WORKERS + 1):
        result = run_benchmark(workers)
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nBenchmark results saved to {OUTPUT_CSV}")

    plot_results(df)