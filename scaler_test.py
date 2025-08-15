import torch 
import numpy as np
from regression_data.data_lims import N_RANGE, K_RANGE, D_RANGE
from models.regression_utils import denormalize_material_params, normalize_material_params

def test_normalizer_denormalizer():
    # Generate test data
    # 1) Exact min
    min_vec = torch.tensor([N_RANGE[0], K_RANGE[0], D_RANGE[0],
                            N_RANGE[0], K_RANGE[0], D_RANGE[0],
                            N_RANGE[0], K_RANGE[0], D_RANGE[0]], dtype=torch.float32)

    # 2) Exact max
    max_vec = torch.tensor([N_RANGE[1], K_RANGE[1], D_RANGE[1],
                            N_RANGE[1], K_RANGE[1], D_RANGE[1],
                            N_RANGE[1], K_RANGE[1], D_RANGE[1]], dtype=torch.float32)

    # 3) Midpoint
    mid_vec = (min_vec + max_vec) / 2

    # 4) Random within range
    rand_vec = torch.empty(9)
    for i in range(0, 9, 3):
        rand_vec[i]   = np.random.uniform(N_RANGE[0], N_RANGE[1])  # n
        rand_vec[i+1] = np.random.uniform(K_RANGE[0], K_RANGE[1])  # k
        rand_vec[i+2] = np.random.uniform(D_RANGE[0], D_RANGE[1])  # d

    # 5) Slightly out of range (to see behaviour)
    out_vec = max_vec + 0.1 * (max_vec - min_vec)

    # Stack into batch
    test_batch = torch.stack([min_vec, max_vec, mid_vec, rand_vec, out_vec], dim=0)

    # Normalize and denormalize
    normed = normalize_material_params(test_batch)
    restored = denormalize_material_params(normed)

    # Compare original and restored
    diff = restored - test_batch
    max_abs_diff = diff.abs().max().item()

    print("Original batch:\n", test_batch)
    print("Normalized:\n", normed)
    print("Restored:\n", restored)
    print(f"Max absolute difference after round-trip: {max_abs_diff:.6e}")

    # Tolerance check
    assert max_abs_diff < 1e-6, "Round-trip failed: difference too large"
    print("✅ Normalizer/denormalizer round-trip test passed!")

# Run the test
test_normalizer_denormalizer()