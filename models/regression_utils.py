import torch
from torch.utils.data import random_split, DataLoader
from regression_data.data_lims import N_RANGE, K_RANGE, D_RANGE
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score
from tabulate import tabulate
from sklearn.preprocessing import MinMaxScaler


# --- Normalizer / Denormalizer ---
# Precompute per-parameter min/max arrays for 9D vector [n1,k1,d1,...,n3,k3,d3]
min_vals = np.array([N_RANGE[0], K_RANGE[0], D_RANGE[0],
                    N_RANGE[0], K_RANGE[0], D_RANGE[0],
                    N_RANGE[0], K_RANGE[0], D_RANGE[0]], dtype=np.float32)

max_vals = np.array([N_RANGE[1], K_RANGE[1], D_RANGE[1],
                    N_RANGE[1], K_RANGE[1], D_RANGE[1],
                    N_RANGE[1], K_RANGE[1], D_RANGE[1]], dtype=np.float32)

# Initialize scaler with feature_range [0,1] and fixed min/max
_SCALER = MinMaxScaler(feature_range=(0, 1))
# Fit on min/max bounds to lock the scaler
_SCALER.fit(np.stack([min_vals, max_vals], axis=0))


def normalize_material_params(params_vector):
    """
    Normalize a 9D material parameter vector using fixed min/max ranges.
    Accepts torch.Tensor of shape [9] or [batch, 9].
    Returns tensor of same shape.
    """
    single_input = False
    if params_vector.ndim == 1:
        params_vector = params_vector.unsqueeze(0)
        single_input = True

    params_np = params_vector.cpu().numpy()
    scaled_np = _SCALER.transform(params_np)
    scaled_tensor = torch.from_numpy(scaled_np).to(params_vector.device).type(params_vector.dtype)

    if single_input:
        return scaled_tensor.squeeze(0)
    return scaled_tensor


def denormalize_material_params(norm_vector):
    """
    Inverse transform of normalized 9D vector back to physical values.
    Accepts torch.Tensor of shape [9] or [batch, 9].
    Returns tensor of same shape.
    """
    single_input = False
    if norm_vector.ndim == 1:
        norm_vector = norm_vector.unsqueeze(0)
        single_input = True

    norm_np = norm_vector.cpu().numpy()
    denorm_np = _SCALER.inverse_transform(norm_np)
    denorm_tensor = torch.from_numpy(denorm_np).to(norm_vector.device).type(norm_vector.dtype)

    if single_input:
        return denorm_tensor.squeeze(0)
    return denorm_tensor

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
    num_samples = int(len(full_dataset))

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

    return train_loader, val_loader, num_samples


def compute_metrics(preds, targets, scaled=True):
    """
    Compute regression metrics for either scaled or unscaled data.
    Returns dict with per-parameter and overall metrics.
    """
    preds_np = preds.numpy()
    targets_np = targets.numpy()
    num_params = preds_np.shape[1]

    rmse = np.sqrt(np.mean((preds_np - targets_np) ** 2, axis=0))
    mae = np.mean(np.abs(preds_np - targets_np), axis=0)
    r2 = [r2_score(targets_np[:, i], preds_np[:, i]) for i in range(num_params)]

    metrics = {
        "RMSE": rmse.tolist(),
        "MAE": mae.tolist(),
        "R2": r2,
        "RMSE_mean": float(np.mean(rmse)),
        "MAE_mean": float(np.mean(mae)),
        "R2_mean": float(np.mean(r2))
    }

    # Extra stats for unscaled domain
    if not scaled:
        # Parameter ranges for normalization
        param_ranges = []
        for i in range(num_params):
            if i % 3 == 0:  # n
                param_ranges.append(N_RANGE[1] - N_RANGE[0])
            elif i % 3 == 1:  # k
                param_ranges.append(K_RANGE[1] - K_RANGE[0])
            else:  # d
                param_ranges.append(D_RANGE[1] - D_RANGE[0])
        param_ranges = np.array(param_ranges)

        nrmse = rmse / param_ranges
        nmae = mae / param_ranges
        max_err = np.max(np.abs(preds_np - targets_np), axis=0)
        p95_err = np.percentile(np.abs(preds_np - targets_np), 95, axis=0)

        metrics.update({
            "NRMSE": nrmse.tolist(),
            "NMAE": nmae.tolist(),
            "MaxError": max_err.tolist(),
            "P95Error": p95_err.tolist()
        })

    return metrics


def test_regression_model(
    model,
    data_loader,
    device,
    loss_fn=None,
    compute_forward_model=False,
    forward_model_fn=None
):
    """
    Evaluate a trained model on a given dataset.
    Computes metrics in both scaled and unscaled domains.
    Optionally computes forward-model reconstruction error.

    Args:
        model: Trained PyTorch model.
        data_loader: DataLoader providing (pulse, normalized_target) pairs.
        device: torch.device.
        loss_fn: Optional loss function (e.g., nn.MSELoss()).
        compute_forward_model: If True, will run forward model for error calc.
        forward_model_fn: Callable mapping unscaled params -> simulated signal.

    Returns:
        metrics: dict with "scaled" and "unscaled" metrics (+forward_model if requested).
        results: dict with raw predictions and targets in both domains.
    """
    model.eval()
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    all_preds_scaled = []
    all_targets_scaled = []
    all_preds_unscaled = []
    all_targets_unscaled = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets_scaled in data_loader:
            inputs = inputs.to(device)
            targets_scaled = targets_scaled.to(device)

            outputs_scaled = model(inputs)

            # Compute loss in scaled space
            total_loss += loss_fn(outputs_scaled, targets_scaled).item() * inputs.size(0)

            # Store scaled
            all_preds_scaled.append(outputs_scaled.cpu())
            all_targets_scaled.append(targets_scaled.cpu())

            # Unscale
            preds_unscaled = denormalize_material_params(outputs_scaled.cpu())
            targets_unscaled = denormalize_material_params(targets_scaled.cpu())

            all_preds_unscaled.append(preds_unscaled)
            all_targets_unscaled.append(targets_unscaled)

    # Stack tensors
    preds_scaled = torch.cat(all_preds_scaled)
    targets_scaled = torch.cat(all_targets_scaled)
    preds_unscaled = torch.cat(all_preds_unscaled)
    targets_unscaled = torch.cat(all_targets_unscaled)

    # Compute metrics
    metrics_scaled = compute_metrics(preds_scaled, targets_scaled, scaled=True)
    metrics_unscaled = compute_metrics(preds_unscaled, targets_unscaled, scaled=False)
    metrics = {
        "scaled": metrics_scaled,
        "unscaled": metrics_unscaled,
        "loss_scaled": total_loss / len(data_loader.dataset)
    }

    # Optional: forward model reconstruction error
    if compute_forward_model and forward_model_fn is not None:
        forward_errors = []
        for pred_params, true_params in zip(preds_unscaled, targets_unscaled):
            sim_pred = forward_model_fn(pred_params)
            sim_true = forward_model_fn(true_params)
            err = torch.sqrt(torch.mean((sim_pred - sim_true) ** 2)).item()
            forward_errors.append(err)
        metrics["forward_model"] = {
            "RMSE": float(np.mean(forward_errors)),
            "RMSE_std": float(np.std(forward_errors))
        }

    # Return metrics and raw data for plotting
    results = {
        "preds_scaled": preds_scaled,
        "targets_scaled": targets_scaled,
        "preds_unscaled": preds_unscaled,
        "targets_unscaled": targets_unscaled
    }

    return metrics, results



def print_metrics_table(metrics, num_layers=1):
    """
    Pretty-print scaled and unscaled metrics for each parameter.
    metrics: dict returned from test_model()
    num_layers: how many layers are in the material stack
    """
    # Build parameter labels
    param_labels = []
    for layer in range(1, num_layers + 1):
        param_labels += [f"n{layer}", f"k{layer}", f"d{layer}"]

    # ----------------
    # Scaled metrics
    # ----------------
    print("\n=== Scaled-domain metrics ===")
    headers = ["Param", "RMSE", "MAE", "R²"]
    table = []
    for i, label in enumerate(param_labels):
        rmse = metrics["scaled"]["RMSE"][i]
        mae = metrics["scaled"]["MAE"][i]
        r2 = metrics["scaled"]["R2"][i]
        table.append([label, f"{rmse:.4f}", f"{mae:.4f}", f"{r2:.4f}"])
    table.append(["Mean", f"{metrics['scaled']['RMSE_mean']:.4f}",
                  f"{metrics['scaled']['MAE_mean']:.4f}",
                  f"{metrics['scaled']['R2_mean']:.4f}"])
    print(tabulate(table, headers=headers, tablefmt="github"))

    # ----------------
    # Unscaled metrics
    # ----------------
    print("\n=== Unscaled-domain metrics ===")
    headers = ["Param", "RMSE", "MAE", "R²", "NRMSE", "MaxErr", "P95Err"]
    table = []
    for i, label in enumerate(param_labels):
        rmse = metrics["unscaled"]["RMSE"][i]
        mae = metrics["unscaled"]["MAE"][i]
        r2 = metrics["unscaled"]["R2"][i]
        nrmse = metrics["unscaled"]["NRMSE"][i]
        maxerr = metrics["unscaled"]["MaxError"][i]
        p95 = metrics["unscaled"]["P95Error"][i]
        table.append([
            label,
            f"{rmse:.4f}", f"{mae:.4f}", f"{r2:.4f}",
            f"{nrmse:.4f}", f"{maxerr:.4f}", f"{p95:.4f}"
        ])
    table.append([
        "Mean",
        f"{metrics['unscaled']['RMSE_mean']:.4f}",
        f"{metrics['unscaled']['MAE_mean']:.4f}",
        f"{metrics['unscaled']['R2_mean']:.4f}",
        f"{np.mean(metrics['unscaled']['NRMSE']):.4f}",
        f"{np.mean(metrics['unscaled']['MaxError']):.4f}",
        f"{np.mean(metrics['unscaled']['P95Error']):.4f}"
    ])
    print(tabulate(table, headers=headers, tablefmt="github"))