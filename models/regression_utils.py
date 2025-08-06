import torch

def convert_target_to_vector(material_params):
    """
    Converts complex n and real d per layer into 9D real vector.

    Args:
        material_params: list of tuples [(n1_complex, d1), (n2, d2), (n3, d3)]

    Returns:
        Tensor of shape (9,) with [n1, k1, d1, ..., n3, k3, d3]
    """
    out = []
    for n_complex, d in material_params:
        n = n_complex.real
        k = -n_complex.imag  # extinction coefficient (positive means loss)
        out.extend([n, k, d])
    return torch.tensor(out, dtype=torch.float32)