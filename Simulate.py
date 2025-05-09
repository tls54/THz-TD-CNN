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


def rts_batched(n0, nj, Dj):
    # All inputs are shape [F]
    c = torch.cos(nj * Dj)
    s = torch.sin(nj * Dj)
    d = c + (0.5j) * (nj / n0 + n0 / nj) * s
    r = (0.5j) * s * (n0 / nj - nj / n0) / d
    t = 1.0 / d
    return r, t * torch.exp(1j * n0 * Dj), t * t - r * r




def RTm_batched(m, n0, layers):
    # Each entry in layers is a tuple of shape [F] tensors: (nj, Dj)
    F = layers[0][0].shape[0]

    U = torch.zeros(F, dtype=torch.cfloat, device=n0.device)
    V = torch.ones(F, dtype=torch.cfloat, device=n0.device)
    T = torch.ones(F, dtype=torch.cfloat, device=n0.device)

    for j in range(m):
        nj, Dj = layers[j]
        r, t, s = rts_batched(n0, nj, Dj)

        Vlast = V.clone()
        U_new = r * V + s * U
        V_new = V - r * U
        U, V = U_new, V_new

        T = T * t * Vlast / V

    R = U / V
    return R, T




# Simulate reference pulse through material, add some noise
def simulate_parallel(x, layers, deltat, noise_level=None):
    L = len(x)
    M = 2 * L
    N = 4 * L
    deltaf = 1.0 / (N * deltat)
    dk = 2 * torch.pi * deltaf / c  

    device = x.device

    # Layer parameters
    indices = torch.stack([
        l[0] if isinstance(l[0], torch.Tensor)
        else torch.tensor(l[0], dtype=torch.cfloat, device=device, requires_grad=True)
        for l in layers
    ])
    thicknesses = torch.stack([
        l[1] if isinstance(l[1], torch.Tensor)
        else torch.tensor(l[1], dtype=torch.cfloat, device=device, requires_grad=True)
        for l in layers
    ])
    m = len(layers)

    # Frequency indices
    k_vals = torch.arange(M + 1, dtype=torch.float32, device=device)
    kD = dk * k_vals[:, None] * thicknesses[None, :]  # Shape: [M+1, m]

    # Build per-frequency layer parameters
    batched_layers = [(indices[j].expand(M+1), kD[:, j]) for j in range(m)]
    
    # Compute batched transmission
    _, T_half = RTm_batched(m, torch.tensor(1.0, dtype=torch.cfloat, device=device), batched_layers)

    # Build full spectrum using conjugate symmetry
    T = torch.zeros(N, dtype=torch.cfloat, device=device)
    T[:M+1] = T_half
    T[M+1:] = torch.conj(torch.flip(T[1:M], dims=[0]))  # Avoid duplicating DC and Nyquist

    # FFT input
    z = torch.zeros(N, dtype=torch.float, device=device)
    z[:L] = x
    X = torch.fft.fft(z) / N

    # Apply transmission
    Y = T * X
    y = N * torch.fft.ifft(Y).real

    if noise_level:
        y += noise_level * torch.randn(N, dtype=torch.float, device=device)

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


