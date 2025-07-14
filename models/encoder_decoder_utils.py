import torch 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels.numpy(), cmap='viridis', s=10, alpha=0.8)
    cbar = plt.colorbar(scatter, ticks=sorted(torch.unique(labels).tolist()))
    cbar.set_label('Number of Layers')
    plt.title(f"Latent Space Visualization with Labels ({method.upper()})")
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()