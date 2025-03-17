import numpy as np
import warp as wp

from ..main import *

sigma = SIGMA
h = SMOOTHING_LENGTH
h_inv = 1.0 / h
h_inv2 = 1.0 / (h * h)

###########################################################################
# smoothing kernels

# FROM: Eqn.(4) of the paper "Smoothed Particle Hydrodynamics Techniques for the Physics Based Simulation of Fluids and Solids"
# REF: https://github.com/InteractiveComputerGraphics/SPH-Tutorial/blob/master/pdf/SPH_Tutorial.pdf


# def spline_W(r: float):
#     q = r / h
#     tmp = 0.0
#     if 1 > q > 0.5:
#         tmp = 2.0 * (1.0 - q) ** 3.0
#         tmp *= sigma
#     elif q <= 0.5:
#         tmp = 6.0 * (q**3.0 - q**2.0) + 1.0
#         tmp *= sigma
#     return tmp


# def grad_spline_W(r: wp.vec3):
#     q = wp.length(r) / h
#     tmp = 0.0
#     if 1 > q > 0.5:
#         tmp = 6.0 * (1.0 - q) ** 2.0
#     elif q <= 0.5:
#         tmp = -6.0 * (3.0 * q**2.0 - 2.0 * q)
#     return tmp * sig_inv_h * wp.normalize(r)


###########################################################################
# precompute kernel values

W_SIZE = 8192
GRAD_W_SIZE = 65536

# Warp arrays for the precomputed tables
W_table = None
grad_W_table = None


def precompute_tables():
    """Create and initialize the precomputed SPH kernel tables"""
    global W_table, grad_W_table

    r_spline = np.linspace(0, h, W_SIZE, dtype=np.float64)
    r_grad = np.linspace(0, h, GRAD_W_SIZE, dtype=np.float64)

    q_spline = r_spline * h_inv
    q_grad = r_grad * h_inv

    # --- Precompute spline_W values ---
    spline_values = np.zeros_like(r_spline)

    mask_low = q_spline <= 0.5
    mask_high = (q_spline > 0.5) & (q_spline < 1.0)

    spline_values[mask_low] = (
        6.0 * (q_spline[mask_low] ** 3 - q_spline[mask_low] ** 2) + 1.0
    ) * sigma
    spline_values[mask_high] = 2.0 * np.power(1.0 - q_spline[mask_high], 3) * sigma

    # --- Precompute grad_spline_W/r values ---
    grad_values = np.zeros_like(r_grad)

    nonzero = r_grad > 1e-10
    mask_low = nonzero & (q_grad <= 0.5)
    mask_high = nonzero & (q_grad > 0.5) & (q_grad < 1.0)

    grad_values[mask_low] = (12.0 - 18.0 * q_grad[mask_low]) * sigma * h_inv2
    grad_values[mask_high] = (
        6.0 * np.power(1.0 - q_grad[mask_high], 2) * sigma * h_inv / r_grad[mask_high]
    )

    # Convert to warp arrays
    W_table = wp.from_numpy(spline_values.astype(np.float32), dtype=wp.float32)
    grad_W_table = wp.from_numpy(grad_values.astype(np.float32), dtype=wp.float32)


precompute_tables()


@wp.func
def spline_W(r: float, table: wp.array(dtype=wp.float32)):  # type: ignore
    if r >= h:
        return 0.0

    # Table lookup with linear interpolation
    idx_float = r * (float(W_SIZE - 1) * h_inv)
    idx_low = wp.min(int(idx_float), W_SIZE - 2)
    alpha = idx_float - float(idx_low)

    return (1.0 - alpha) * table[idx_low] + alpha * table[idx_low + 1]


@wp.func
def grad_spline_W(r: wp.vec3, table: wp.array(dtype=wp.float32)):  # type: ignore
    r_len = wp.length(r)

    if r_len >= h or r_len < 1e-10:
        return wp.vec3(0.0, 0.0, 0.0)

    # Table lookup with linear interpolation
    idx_float = r_len * (float(GRAD_W_SIZE - 1) * h_inv)
    idx_low = wp.min(int(idx_float), GRAD_W_SIZE - 2)
    alpha = idx_float - float(idx_low)

    scalar = (1.0 - alpha) * table[idx_low] + alpha * table[idx_low + 1]
    return scalar * r
