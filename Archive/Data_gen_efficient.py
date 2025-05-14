import torch
import numpy as np
import concurrent.futures
from tqdm import tqdm
from Simulate import simulate_parallel, simulate_reference
from time import perf_counter

# Multi-processing employed to speed up this process, CPU intensive task, more cores / workers = more faster (to an extent). 
# Info on the multi-processing methods can be found here: https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor



L = 2**12

# Define utility functions
def downsample_tensor(tensor, factor):
    """Downsample the given tensor by an integer factor."""
    return tensor[::factor]  # Simple downsampling

# create set of num layers -> int for each sample is num layers in sample
def generate_samples():
    """Generate the number of layers for each sample."""
    return np.random.randint(LAYER_LIMS[0], LAYER_LIMS[1] + 1, NUM_SAMPLES)

# generates random material parameters for 
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
    downsampled_pulse = downsample_tensor(pulse, 4)
    return downsampled_pulse, material_samples[idx], len(material_samples[idx])

def process_sample_wrapper(args):
    """Unpack arguments and call process_sample."""
    idx, reference_pulse, material_samples, deltat = args
    return process_sample(idx, reference_pulse, material_samples, deltat)

def main():
    """Main function to generate, process, and save synthetic data."""
    print("Generating synthetic dataset...")

    # Generate number of layers per sample
    samples = generate_samples()
    total_layers = sum(samples)

    print(f"Total layers to generate: {total_layers}")

    # Generate material parameters
    n_values, k_values, D_values = generate_material_parameters(total_layers)

    # Construct material samples
    material_samples = construct_material_samples(samples, n_values, k_values, D_values)

    # Generate reference pulse
    deltat = 0.0194e-12  # Time step
    L = 2**12  # Number of time points = 4096
    reference_pulse = simulate_reference(L, deltat)

    print("Processing samples in parallel...")
    num_workers = 1  # Adjust based on available cores

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:   # Assigns an interpreter to each process to increase speed.
        results = list(tqdm(
            executor.map(process_sample_wrapper, 
                        [(idx, reference_pulse, material_samples, deltat) for idx in range(NUM_SAMPLES)]),
            total=NUM_SAMPLES,
            ncols=80
        ))
    print("Processing complete. Saving dataset...")

    # Extract processed data
    synthetic_data, material_params, num_layers = zip(*results)

    # Convert to tensors
    synthetic_data = torch.stack(synthetic_data)  # Shape: (NUM_SAMPLES, downsampled_length)
    num_layers = torch.tensor(num_layers)  # Shape: (NUM_SAMPLES,)

    # Save the dataset
    torch.save({
        "synthetic_data": synthetic_data,
        "material_params": material_params,
        "num_layers": num_layers
    }, "synthetic_data.pt")

    print("Dataset saved successfully as synthetic_data.pt")

if __name__ == "__main__":

    # Define limits for number of layers
    LAYER_LIMS = [1, 3]
    NUM_SAMPLES = 1000

    # Define material parameter ranges
    N_RANGE = (1.1, 6.0)
    K_RANGE = (-0.1, 0.01)
    D_RANGE = (0.05e-3, 0.5e-3)

    num_workers = 32
    print('Running new version of data generation script')
    start = perf_counter()
    main()
    end = perf_counter()

    print(f'Time for 1000 samples with {num_workers} cores: {end - start}s')