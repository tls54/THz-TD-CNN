import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from Simulate import simulate_parallel, simulate_reference
from time import perf_counter
from math import ceil

L = 2**12  # Time points
DOWNSAMPLE_FACTOR = 4
DELTA_N_THRESHOLD = 0.5


workers = cpu_count()

def downsample_tensor(tensor, factor):
    return tensor[::factor]

def generate_samples():
    return np.random.randint(LAYER_LIMS[0], LAYER_LIMS[1] + 1, NUM_SAMPLES)

def generate_material_parameters(samples, delta_threshold, verbose=False):
    total_layers = sum(samples)
    n_values = []
    k_values = []
    D_values = []

    total_regenerations = 0
    layer_index = 0

    for sample_idx, num_layers in enumerate(samples):
        sample_n = []
        for i in range(num_layers):
            valid = False
            attempts = 0
            while not valid:
                n_new = np.random.uniform(*N_RANGE)
                if i == 0 or abs(n_new - sample_n[-1]) >= delta_threshold:
                    valid = True
                else:
                    total_regenerations += 1
                    if verbose:
                        print(f"[Sample {sample_idx}, Layer {i}] Δn = {abs(n_new - sample_n[-1]):.4f} < {delta_threshold} — regenerating...")
                attempts += 1
                if attempts > 100:
                    raise RuntimeError(f"Too many retries: Δn condition failed at sample {sample_idx}, layer {i}")

            sample_n.append(n_new)
            n_values.append(n_new)
            k_values.append(np.random.uniform(*K_RANGE))
            D_values.append(np.random.uniform(*D_RANGE))

            layer_index += 1

    if verbose:
        print(f"Total Δn violations (regenerations): {total_regenerations}")
    else:
        print(f"Total regenerations due to Δn: {total_regenerations}")
        
    return np.array(n_values), np.array(k_values), np.array(D_values)



def validate_delta_n_spacing(material_samples, delta_threshold, verbose=True):
    violations = 0
    for sample_idx, sample in enumerate(material_samples):
        for i in range(len(sample) - 1):
            n1 = sample[i][0].real
            n2 = sample[i + 1][0].real
            delta_n = abs(n2 - n1)
            if delta_n < delta_threshold:
                if verbose:
                    print(f"[Validation Fail] Sample {sample_idx}, Layers {i}-{i+1}: Δn = {delta_n:.4f} < {delta_threshold}")
                violations += 1
    if violations == 0:
        print("✅ All samples passed Δn spacing validation.")
    else:
        print(f"❌ {violations} violations found in Δn spacing.")



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
    """Function run in each process to handle a batch of samples."""
    reference_pulse, material_samples_batch, indices, deltat = args
    batch_results = []
    for i, idx in enumerate(indices):
        pulse = simulate_parallel(reference_pulse, material_samples_batch[i], deltat, 0)[1].detach().cpu()[:L]
        downsampled_pulse = downsample_tensor(pulse, DOWNSAMPLE_FACTOR)
        batch_results.append((downsampled_pulse, material_samples_batch[i], len(material_samples_batch[i])))
    return batch_results



def main():
    print("Generating synthetic dataset (multi-process batched)...")

    samples = generate_samples()
    total_layers = sum(samples)
    print(f"Total layers to generate: {total_layers}")

    n_values, k_values, D_values = generate_material_parameters(samples, DELTA_N_THRESHOLD)
    material_samples = construct_material_samples(samples, n_values, k_values, D_values)

    deltat = 0.0194e-12
    reference_pulse = simulate_reference(L, deltat)

    # --- Prepare batches ---
    NUM_PROCESSES = min(workers, 32)
    batch_size = ceil(len(samples) / NUM_PROCESSES)

    sample_batches = [samples[i*batch_size : min((i+1)*batch_size, len(samples))] for i in range(NUM_PROCESSES)]
    material_batches = [material_samples[i*batch_size : min((i+1)*batch_size, len(material_samples))] for i in range(NUM_PROCESSES)]
    index_batches = [list(range(i*batch_size, min((i+1)*batch_size, len(samples)))) for i in range(NUM_PROCESSES)]

    args = [(reference_pulse, material_batches[i], index_batches[i], deltat) for i in range(NUM_PROCESSES)]

    print(f"Processing {NUM_SAMPLES} samples across {NUM_PROCESSES} processes...")

    with Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap_unordered(process_batch, args), total=NUM_PROCESSES, ncols=80))

    print("Processing complete. Aggregating results...")

    all_results = [item for sublist in results for item in sublist]

    assert len(all_results) == NUM_SAMPLES, f"Expected {NUM_SAMPLES} samples, got {len(all_results)}"

    synthetic_data, material_params, num_layers = zip(*all_results)

    validate_delta_n_spacing(material_samples, DELTA_N_THRESHOLD)

    synthetic_data = torch.stack(synthetic_data)
    num_layers = torch.tensor(num_layers)

    torch.save({
        "synthetic_data": synthetic_data,
        "material_params": material_params,
        "num_layers": num_layers
    }, "synthetic_data.pt")

    print("Dataset saved successfully as synthetic_data.pt")

if __name__ == "__main__":
    LAYER_LIMS = [1, 3]
    NUM_SAMPLES = 10_000

    N_RANGE = (1.1, 6.0)
    K_RANGE = (-0.1, 0.001)
    D_RANGE = (0.05e-3, 0.5e-3)

    print("Running multiprocessing batched data generation script")
    
    start = perf_counter()
    main()
    end = perf_counter()

    print(f'Time for {NUM_SAMPLES} samples (multi-process batched): {end - start:.2f}s')