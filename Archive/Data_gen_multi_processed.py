import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from Simulate import simulate_parallel, simulate_reference
from time import perf_counter

L = 2**12  # Time points
DOWNSAMPLE_FACTOR = 4

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

    n_values, k_values, D_values = generate_material_parameters(total_layers)
    material_samples = construct_material_samples(samples, n_values, k_values, D_values)

    deltat = 0.0194e-12
    reference_pulse = simulate_reference(L, deltat)

    # --- Prepare batches ---
    NUM_PROCESSES = min(cpu_count(), 32)  # Max out logical threads
    print(f'CPU Count: {cpu_count()}')
    batch_size = len(samples) // NUM_PROCESSES
    sample_batches = [samples[i*batch_size:(i+1)*batch_size] for i in range(NUM_PROCESSES)]
    material_batches = [material_samples[i*batch_size:(i+1)*batch_size] for i in range(NUM_PROCESSES)]
    index_batches = [list(range(i*batch_size, (i+1)*batch_size)) for i in range(NUM_PROCESSES)]

    args = [(reference_pulse, material_batches[i], index_batches[i], deltat) for i in range(NUM_PROCESSES)]

    print(f"Processing {NUM_SAMPLES} samples across {NUM_PROCESSES} processes...")

    with Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap_unordered(process_batch, args), total=NUM_PROCESSES, ncols=80))

    print("Processing complete. Aggregating results...")

    all_results = [item for sublist in results for item in sublist]
    synthetic_data, material_params, num_layers = zip(*all_results)

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
    NUM_SAMPLES = 60_000

    N_RANGE = (1.1, 6.0)
    K_RANGE = (-0.1, 0.01)
    D_RANGE = (0.05e-3, 0.5e-3)



    print("Running multiprocessing batched data generation script")
    start = perf_counter()
    main()
    end = perf_counter()

    print(f'Time for {NUM_SAMPLES} samples (multi-process batched): {end - start:.2f}s')