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

    c = torch.cos(nj * Dj)  # Cosine of phase shift
    s = torch.sin(nj * Dj)  # Sine of phase shift
    d = c + (0.5j) * (nj / n0 + n0 / nj) * s  # Denominator term
    r = (0.5j) * s * (n0 / nj - nj / n0) / d  # Reflection coefficient
    t = 1.0 / d  # Transmission coefficient
    return r, t * torch.exp(1j * n0 * Dj), t * t - r * r




def RTm(m, n0, layers):
    """
    Computes the overall reflection (R) and transmission (T) coefficients 
    for a stack of m layers using the transfer matrix method.

    Args:
        m (int): Number of layers.
        n0 (torch.Tensor): Refractive index of the incident medium.
        layers (list of tuples): Each tuple contains:
            - `nj` (torch.Tensor): Refractive index of the j-th layer.
            - `Dj` (torch.Tensor): Thickness of the j-th layer.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
        - R: Total reflection coefficient of the stack.
        - T: Total transmission coefficient of the stack.

    The function iteratively applies the transfer matrix method by combining 
    the effects of multiple layers, each contributing a phase shift and partial 
    reflection/transmission. The final result gives the net reflection and 
    transmission of the multi-layer system.
    """

    U = torch.tensor(0.0, dtype=torch.cfloat)
    V = torch.tensor(1.0, dtype=torch.cfloat)
    T = torch.tensor(1.0, dtype=torch.cfloat)
    R = torch.tensor(0.0, dtype=torch.cfloat)

    for j in range(m):
        nj, Dj = layers[j]  # Extract layer properties
        r, t, s = rts(n0, nj, Dj)  # Get layer coefficients

        Vlast = V.clone()  # Store previous V
        U, V = r * V + s * U, V - r * U  # Update U and V

        T = T * t * Vlast / V  # Update transmission coefficient
        R = U / V  # Compute reflection coefficient

    return R, T




# Simulate reference pulse through material, add some noise
def simulate_parallel(x, layers, deltat, noise_level=None):
    """
    Simulates the propagation of a reference pulse through a layered medium using 
    the transfer matrix method and Fourier domain computations.

    Args:
        x (torch.Tensor): Time-domain reference pulse (1D array).
        layers (list of tuples): Each tuple contains:
            - `nj` (torch.Tensor or float): Refractive index of the j-th layer.
            - `Dj` (torch.Tensor or float): Thickness of the j-th layer.
        deltat (float): Time step between samples in the reference pulse.
        noise_level (float, optional): Standard deviation of Gaussian noise.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
        - T: Transmission coefficients in the frequency domain.
        - y: Simulated time-domain signal after propagation.
    """
    
    L = len(x)

    # Define FFT sizes
    M = 6 * L
    N = 12 * L

    # Wave number increment per frequency step
    deltaf = 1.0 / (N * deltat)
    dk = 2 * torch.pi * deltaf / c  

    # Convert layer parameters into tensors, preserving autograd tracking
    device = x.device  # Ensure compatibility with input tensor
    indices = torch.stack([l[0] if isinstance(l[0], torch.Tensor) else torch.tensor(l[0], dtype=torch.cfloat, device=device, requires_grad=True) for l in layers])

    thicknesses = torch.stack([l[1] if isinstance(l[1], torch.Tensor) else torch.tensor(l[1], dtype=torch.cfloat, device=device, requires_grad=True) for l in layers])

    m = len(layers)

    # Initialize transmission coefficients array
    T = torch.zeros(N, dtype=torch.cfloat, device=device)

    # Compute frequency-dependent transmission coefficients
    for k in range(M + 1):
        kD = k * dk * thicknesses  # Compute phase shifts per layer
        nkD = [(indices[l], kD[l]) for l in range(m)]  # Refractive indices with phase shifts
        T[k] = RTm(m, torch.tensor(1.0, dtype=torch.cfloat, device=device), nkD)[1]  

    # Ensure symmetry in Fourier domain
    for k in range(1, M):  
        T[N - k] = torch.conj(T[k])

    # Zero-pad the input signal
    z = torch.zeros(N, dtype=torch.float, device=device)
    z[:L] = x[:L]

    # Perform FFT on the input signal
    X = torch.fft.fft(z) / N

    # Apply transmission coefficients in the frequency domain
    Y = T * X

    # Inverse FFT to get back to time domain
    y = N * torch.fft.ifft(Y).real

    # Add Gaussian noise if specified
    if noise_level:
        y += noise_level * torch.randn(N, dtype=torch.float32, device=device)

    return T, y