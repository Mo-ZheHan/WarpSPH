import warp as wp

###########################################################################
# Global variable settings
GRAVITY = 9.8
DIAMETER = 0.05
OMEGA = 0.5
ETA = 1.0e-3
RHO_0 = 1.0e3
INV_SMALL = 1.0e-6

# Scene settings
BOX_WIDTH = 1.0
BOX_HEIGHT = 1.0
BOX_LENGTH = 1.0

###########################################################################
# Computed intermediate variables
SMOOTHING_LENGTH = wp.constant(2 * DIAMETER)
TIME_STEP_MAX = wp.constant(5e-3 * DIAMETER)
FLUID_MASS = wp.constant(RHO_0 * DIAMETER**3)
SIGMA = wp.constant(8 / (wp.pi * SMOOTHING_LENGTH**3))
VIS_MU = wp.constant(1.0e-3 * RHO_0)
