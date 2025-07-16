import torch 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch import nn as nn



def test_model(index, model, device, time_pulses, deltat):
    real_pulse = time_pulses[index]  
    input_pulse = real_pulse.to(device)#.unsqueeze(0)

    L = time_pulses.shape[1]
    t_axis = np.arange(0, L*deltat, deltat)

    model.eval()
    with torch.no_grad():
        recon_pulse = model(input_pulse)

    # Bring both to CPU and remove batch dim if added
    recon_pulse_cpu = recon_pulse.squeeze(0).cpu()
    real_pulse_cpu = real_pulse.cpu()


    criterion = nn.MSELoss()
    loss = criterion(recon_pulse_cpu, real_pulse_cpu).item()
    print(f'Loss between signals: {loss}')

    # Convert to NumPy for plotting
    recon_np = recon_pulse_cpu.numpy()
    real_np = real_pulse_cpu.numpy()

    plt.figure(figsize=(12,4))
    plt.title('Example THz Time Domain pulse')
    plt.plot(t_axis*1e12, real_np, label='Input Pulse')
    plt.plot(t_axis*1e12, recon_np, label='Reconstructed Pulse')
    plt.ylabel('Signal')
    plt.xlabel('Time, t [ps]')
    plt.legend()
    plt.show()



def plot_latent_space(model, dataloader, device, method='tsne', max_samples=1000):
    model.eval()
    model.to(device)

    latents = []
    with torch.no_grad():
        for x_batch in dataloader:
            x_batch = x_batch[0].to(device)
            z = model.encoder(x_batch)
            latents.append(z.cpu())

    latents = torch.cat(latents, dim=0)

    # Randomly subsample if needed
    if max_samples is not None and latents.shape[0] > max_samples:
        indices = np.random.choice(latents.shape[0], size=max_samples, replace=False)
        latents = latents[indices]

    # Choose dimensionality reduction method
    if latents.shape[1] > 2:
        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=30, learning_rate=200, init='pca')
        elif method == 'pca':
            reducer = PCA(n_components=2)
        else:
            raise ValueError("method should be 'tsne' or 'pca'")
        latents_2d = reducer.fit_transform(latents.numpy())
    else:
        latents_2d = latents.numpy()

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], s=5, alpha=0.7)
    plt.title(f"Latent Space Visualization ({method.upper()})")
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return latents


def plot_latent_space_with_labels(model, synthetic_data, labels, device, method='tsne', max_samples=1000):
    model.eval()
    model.to(device)

    with torch.no_grad():
        synthetic_data = synthetic_data.to(device)
        latents = model.encoder(synthetic_data).cpu()

    labels = labels.cpu()

    # Subsample randomly if needed
    if max_samples is not None and latents.shape[0] > max_samples:
        indices = torch.randperm(latents.shape[0])[:max_samples]
        latents = latents[indices]
        labels = labels[indices]

    # Dimensionality reduction
    if latents.shape[1] > 2:
        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=30, learning_rate=200, init='pca')
        elif method == 'pca':
            reducer = PCA(n_components=2)
        else:
            raise ValueError("method should be 'tsne' or 'pca'")
        latents_2d = reducer.fit_transform(latents.numpy())
    else:
        latents_2d = latents.numpy()

    # Plot
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels.numpy(), cmap='viridis', s=7, alpha=0.8)
    cbar = plt.colorbar(scatter, ticks=sorted(torch.unique(labels).tolist()))
    cbar.set_label('Number of Layers')
    plt.title(f"Latent Space Visualization with Labels ({method.upper()})")
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()