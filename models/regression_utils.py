import torch
from torch.utils.data import random_split, DataLoader
from regression_data.data_lims import N_RANGE, K_RANGE, D_RANGE

# --- Normalizer / Denormalizer ---

def normalize_material_params(params_vector):
    """Min-max normalize 9D vector [n1, k1, d1, ..., n3, k3, d3]."""
    n_min, n_max = N_RANGE
    k_min, k_max = K_RANGE
    d_min, d_max = D_RANGE

    norm = params_vector.clone()
    norm[0::3] = (norm[0::3] - n_min) / (n_max - n_min)  # n
    norm[1::3] = (norm[1::3] - k_min) / (k_max - k_min)  # k
    norm[2::3] = (norm[2::3] - d_min) / (d_max - d_min)  # d
    return norm


def denormalize_material_params(norm_vector):
    """Inverse transform of normalized 9D vector back to physical values."""
    n_min, n_max = N_RANGE
    k_min, k_max = K_RANGE
    d_min, d_max = D_RANGE

    denorm = norm_vector.clone()
    denorm[0::3] = denorm[0::3] * (n_max - n_min) + n_min
    denorm[1::3] = denorm[1::3] * (k_max - k_min) + k_min
    denorm[2::3] = denorm[2::3] * (d_max - d_min) + d_min
    return denorm

# --- Vectorize raw target from dataset ---

def convert_target_to_vector(material_params_entry):
    """
    Convert [(n_complex, d), ...] to 9D vector: [n1, k1, d1, n2, k2, d2, ...]
    """
    out = []
    for n_complex, d in material_params_entry:
        n = n_complex.real
        k = n_complex.imag  # extinction coefficient (-ve means loss)
        out.extend([n, k, d])
    return torch.tensor(out, dtype=torch.float32)

# --- Dataset loader for regression ---

def load_regression_dataset(path):
    """
    Load .pt dataset and prepare (pulse, normalized target) pairs.

    Returns:
        A list of (pulse_tensor, normalized_target_tensor)
    """
    print(f"Loading data from {path}")
    data = torch.load(path, weights_only=False)

    synthetic_data = data["synthetic_data"]      # shape: (N, 1024)
    material_params = data["material_params"]    # length N, list of 3-layer specs

    dataset = []
    for pulse, raw_params in zip(synthetic_data, material_params):
        target_vec = convert_target_to_vector(raw_params)
        norm_target = normalize_material_params(target_vec)
        pulse_tensor = pulse.unsqueeze(0)  # add channel dim: [1, 1024]
        dataset.append((pulse_tensor, norm_target))

    print("Dataset loaded successfully!")
    print(f"Number of samples: {len(dataset)}")
    print(f"Shape of input pulse: {dataset[0][0].shape}, target vector: {dataset[0][1].shape}")
    return dataset




def get_train_val_loaders(
    dataset_path,
    batch_size=64,
    val_split=0.1,
    shuffle=True,
    seed=42,
    num_workers=0,  # set >0 if using multiprocessing
    pin_memory=True,
):
    full_dataset = load_regression_dataset(dataset_path)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader