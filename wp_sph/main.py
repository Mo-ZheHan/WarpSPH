import warp as wp

###########################################################################
# Global variable settings

GRAVITY = 0.1
SMOOTHING_LENGTH = 0.8  # NOTE change this to adjust number of particles


###########################################################################
# Computed intermediate variables
SIGMA = 8 / (wp.pi * SMOOTHING_LENGTH**3)
SIG_INV_H = SIGMA / SMOOTHING_LENGTH
FLUID_MASS = 0.01 * SMOOTHING_LENGTH**3  # reduce according to smoothing length
