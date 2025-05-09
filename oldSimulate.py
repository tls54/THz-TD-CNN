import numpy as np
import torch

c = 299792458 

# Create a reference pulse
def simulate_reference(L, deltat):
    toff = 1.0e-11
    twidth = 8.0e-13
    tdecay = 1.0e-12
    scale = 1.0e12

    t = torch.arange(0, L, dtype=torch.float32) * deltat - toff
    x = -scale * t * torch.exp(-(t / twidth) ** 2 - t / tdecay)

    return x


def rts(n0, nj, Dj):
    """
    Computes the reflection and transmission coefficients for a single layer.

    Args:
        n0 (torch.Tensor): Refractive index of the incident medium.
        nj (torch.Tensor): Refractive index of the layer.
        Dj (torch.Tensor): Thickness of the layer.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        - r: Reflection coefficient.
        - t: Transmission coefficient (with phase shift applied).
        - s: Quantity t**2 - r**2, used in further calculations.

    The function calculates wave propagation effects through a thin layer 
    based on the transfer matrix method. It includes phase shifts due to 
    propagation through the layer and Fresnel coefficients for reflections.
    """

    c = torch.cos(nj * Dj)  # Cosine of phase shift (nj Dj is optical path)
    s = torch.sin(nj * Dj)  # Sine of phase shift
    d = c + (0.5j) * (nj / n0 + n0 / nj) * s  # Denominator term
    r = (0.5j) * s * (n0 / nj - nj / n0) / d  # Reflection coefficient
    t = 1.0 / d  # Transmission coefficient
    return r, t * torch.exp(1j * n0 * Dj), t * t - r * r




def RTm_batch(n0, nj, Dj):
    """
    Batched version of RTm.
    Args:
        n0: complex scalar (incident index)
        nj: Tensor of shape (m,) - refractive indices of each layer
        Dj: Tensor of shape (K, m) - phase shifts per frequency per layer

    Returns:
        T: Tensor of shape (K,) - frequency-dependent transmission coefficient
    """
    K, m = Dj.shape
    dtype = torch.cfloat
    device = Dj.device

    U = torch.zeros(K, dtype=dtype, device=device)
    V = torch.ones(K, dtype=dtype, device=device)
    T = torch.ones(K, dtype=dtype, device=device)

    for j in range(m):
        njj = nj[j]  # Scalar complex
        Dj_j = Dj[:, j]  # Shape (K,)

        c = torch.cos(njj * Dj_j)
        s = torch.sin(njj * Dj_j)
        d = c + 0.5j * (njj / n0 + n0 / njj) * s
        r = 0.5j * s * (n0 / njj - njj / n0) / d
        t = 1.0 / d

        Vlast = V.clone()
        U, V = r * V + (t * t - r * r) * U, V - r * U
        T = T * t * Vlast / V

    return T


# Simulate reference pulse through material, add some noise
def simulate_parallel(x, layers, deltat, noise_level=None):
    """
    Simulates the time-domain transmission of a THz pulse through a multilayer material system using
    the transfer matrix method, fully vectorized over frequency components.

    Args:
        x (Tensor): Real input pulse in the time domain, shape (L,).
        layers (list of tuples): Each tuple represents a layer and is of the form (n, d),
            where:
                - n is a complex refractive index (can be a scalar or torch.tensor).
                - d is the thickness in meters (can be a scalar or torch.tensor).
        deltat (float): Time step between samples in the time-domain pulse (in seconds).
        noise_level (float, optional): Standard deviation of additive Gaussian noise
            to be added to the output signal. If None, no noise is added.

    Returns:
        T (Tensor): Frequency-domain transmission coefficients, shape (N,), complex-valued.
        y (Tensor): Simulated real-valued time-domain transmitted signal, shape (N,).
    
    Notes:
        - The input signal x is zero-padded to length N = 4 * L for FFT-based convolution.
        - Transmission coefficients are computed for all positive frequencies using a batched
        implementation of the transfer matrix method, then extended to full spectrum
        via Hermitian symmetry to ensure a real-valued time-domain output.
        - Designed for efficient use on GPU and differentiable with PyTorch autograd.
    """


    L = len(x)
    M = 2 * L
    N = 4 * L

    device = x.device
    dtype = torch.cfloat

    deltaf = 1.0 / (N * deltat)
    dk = 2 * torch.pi * deltaf / c
    K = M + 1  # Number of forward frequency samples

    # Convert to tensor and ensure gradient tracking
    indices = torch.stack([
        l[0] if isinstance(l[0], torch.Tensor) else torch.tensor(l[0], dtype=dtype, device=device, requires_grad=True)
        for l in layers
    ])
    thicknesses = torch.stack([
        l[1] if isinstance(l[1], torch.Tensor) else torch.tensor(l[1], dtype=dtype, device=device, requires_grad=True)
        for l in layers
    ])
    m = len(layers)

    # Vectorized kD: shape (K, m)
    k_vals = torch.arange(0, K, dtype=torch.float32, device=device)
    Dj = dk * k_vals[:, None] * thicknesses[None, :]  # (K, m)

    # Compute T across all frequencies
    n0 = torch.tensor(1.0, dtype=dtype, device=device)
    T_forward = RTm_batch(n0, indices, Dj)  # shape (K,)

    # Build full T of length N using Hermitian symmetry
    T = torch.zeros(N, dtype=dtype, device=device)
    T[:K] = T_forward
    T[-(M-1):] = torch.conj(T[1:M])  # Hermitian symmetry

    # Pad signal and FFT
    z = torch.zeros(N, dtype=torch.float32, device=device)
    z[:L] = x
    X = torch.fft.fft(z) / N

    Y = T * X
    y = N * torch.fft.ifft(Y).real

    if noise_level:
        y += noise_level * torch.randn(N, dtype=torch.float32, device=device)

    return T, y


if __name__ == '__main__':
    import time

    # Simulation parameters
    L = 2**12
    deltat = 0.0194e-12

    # Generate reference pulse
    x = simulate_reference(L, deltat)

    # Define a 3-layer system
    layers = [
        (torch.tensor(3.5 + 0.001j, dtype=torch.cfloat), torch.tensor(2e-3, dtype=torch.cfloat)),
        (torch.tensor(2.8 + 0.0005j, dtype=torch.cfloat), torch.tensor(1.5e-3, dtype=torch.cfloat)),
        (torch.tensor(4.0 + 0.002j, dtype=torch.cfloat), torch.tensor(1.0e-3, dtype=torch.cfloat)),
    ]

    # Time the function
    t0 = time.perf_counter()
    T, y = simulate_parallel(x, layers, deltat)
    t1 = time.perf_counter()

    print(f"simulate_parallel execution time: {t1 - t0:.4f} seconds")


