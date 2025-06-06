import torch
import numpy as np
from tqdm import tqdm
from Simulate import simulate_parallel, simulate_reference
from time import perf_counter



L = 2**12  # Time points
DOWNSAMPLE_FACTOR = 4

def downsample_tensor(tensor, factor):
    """Downsample the given tensor by an integer factor."""
    return tensor[::factor]

def generate_samples():
    """Generate the number of layers for each sample."""
    return np.random.randint(LAYER_LIMS[0], LAYER_LIMS[1] + 1, NUM_SAMPLES)

def generate_material_parameters(total_layers):
    """Generate random material parameters for all layers."""
    n_values = np.random.uniform(*N_RANGE, total_layers)
    k_values = np.random.uniform(*K_RANGE, total_layers)
    D_values = np.random.uniform(*D_RANGE, total_layers)
    return n_values, k_values, D_values

def construct_material_samples(samples, n_values, k_values, D_values):
    """Construct material samples from generated parameters."""
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

def process_sample(idx, reference_pulse, material_samples, deltat):
    """Process a single sample for pulse propagation and downsampling."""
    pulse = simulate_parallel(reference_pulse, material_samples[idx], deltat, 0)[1].detach().cpu()[:L]
    downsampled_pulse = downsample_tensor(pulse, DOWNSAMPLE_FACTOR)
    return downsampled_pulse, material_samples[idx], len(material_samples[idx])

def main():
    print("Generating synthetic dataset (single-threaded)...")

    samples = generate_samples()
    total_layers = sum(samples)
    print(f"Total layers to generate: {total_layers}")

    n_values, k_values, D_values = generate_material_parameters(total_layers)
    material_samples = construct_material_samples(samples, n_values, k_values, D_values)

    deltat = 0.0194e-12
    reference_pulse = simulate_reference(L, deltat)

    print("Processing samples one-by-one (no multiprocessing)...")
    results = []
    for idx in tqdm(range(NUM_SAMPLES), ncols=80):
        results.append(process_sample(idx, reference_pulse, material_samples, deltat))

    print("Processing complete. Saving dataset...")

    synthetic_data, material_params, num_layers = zip(*results)

    synthetic_data = torch.stack(synthetic_data)
    num_layers = torch.tensor(num_layers)

    torch.save({
        "synthetic_data": synthetic_data,
        "material_params": material_params,
        "num_layers": num_layers
    }, "synthetic_data.pt")

    print("Dataset saved successfully as synthetic_data.pt")

if __name__ == "__main__":

    # Define limits for number of layers
    LAYER_LIMS = [1, 3]
    NUM_SAMPLES = int(60_000)


    # Define material parameter ranges
    N_RANGE = (1.1, 6.0)
    K_RANGE = (-0.1, 0.01)
    D_RANGE = (0.05e-3, 0.5e-3)

    print('Running single-threaded version of data generation script')
    start = perf_counter()
    main()
    end = perf_counter()

    print(f'Time for {NUM_SAMPLES} samples (single-threaded): {end - start:.2f}s')