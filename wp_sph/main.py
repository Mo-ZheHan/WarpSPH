import warp as wp

###########################################################################
# Global variable settings

GRAVITY = 9.8
SMOOTHING_LENGTH = 0.8  # NOTE change this to adjust number of particles
OMEGA = 0.5
RHO_0 = 1.0
ETA = 1.0e-4
INV_SMALL = 1.0e-6


###########################################################################
# Computed intermediate variables
SIGMA = 8 / (wp.pi * SMOOTHING_LENGTH**3)
SIG_INV_H = SIGMA / SMOOTHING_LENGTH
FLUID_MASS = 0.3 * SMOOTHING_LENGTH**3  # reduce according to smoothing length
