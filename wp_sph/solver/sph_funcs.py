import warp as wp

from ..main import *

h = SMOOTHING_LENGTH
sigma = SIGMA
sig_inv_h = SIG_INV_H

###########################################################################
# smoothing kernels

# FROM: Eqn.(4) of the paper "Smoothed Particle Hydrodynamics Techniques for the Physics Based Simulation of Fluids and Solids"
# REF: https://github.com/InteractiveComputerGraphics/SPH-Tutorial/blob/master/pdf/SPH_Tutorial.pdf


@wp.func
def spline_W(r: float):
    q = r / h
    tmp = 0.0
    if 1 > q > 0.5:
        tmp = 2.0 * (1.0 - q) ** 3.0
        tmp *= sigma
    elif q <= 0.5:
        tmp = 6.0 * (q**3.0 - q**2.0) + 1.0
        tmp *= sigma
    return tmp


@wp.func
def grad_spline_W(r: wp.vec3):
    q = wp.length(r) / h
    tmp = 0.0
    if 1 > q > 0.5:
        tmp = -6.0 * (1.0 - q) ** 2.0
    elif q <= 0.5:
        tmp = 6.0 * (3.0 * q**2.0 - 2.0 * q)
    return tmp * sig_inv_h * wp.normalize(r)
