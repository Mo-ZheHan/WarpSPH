from enum import Enum
from pathlib import Path

import warp as wp

MODEL_DIR = Path(__file__).parents[1] / "models"


def model_dir(filename):
    return MODEL_DIR / filename


class SceneType(Enum):
    HOUSE = False  # if the rigid body is static
    PLANE = True  # if the rigid body is dynamic
    HAND = True

scene_type = SceneType.HAND
DYNAMIC_SCENE = scene_type.value


###########################################################################
# Global variable settings
GRAVITY = 9.8
DIAMETER = 0.05
OMEGA = 0.5
ETA = 1.0e-3
RHO_0 = 1.0e3
INV_SMALL = 1.0e-6
FPS = 60

###########################################################################
# Scene settings
if scene_type == SceneType.HOUSE:
    BOX_WIDTH = 8.0
    BOX_HEIGHT = 10.0
    BOX_LENGTH = 12.0
elif scene_type == SceneType.PLANE:
    BOX_WIDTH = 8.0
    BOX_HEIGHT = 8.0
    BOX_LENGTH = 8.0
    GRAVITY = 0.0  # No gravity for plane scene
elif scene_type == SceneType.HAND:
    BOX_WIDTH = 20.0
    BOX_HEIGHT = 80.0
    BOX_LENGTH = 20.0
    GRAVITY /= 2.0
else:
    raise ValueError(f"Unknown scene type: {scene_type}")

###########################################################################
# Computed intermediate variables
SMOOTHING_LENGTH = wp.constant(2 * DIAMETER)
TIME_STEP_MAX = wp.constant(5e-2 * DIAMETER)
FLUID_MASS = wp.constant(RHO_0 * DIAMETER**3)
SIGMA = wp.constant(8 / (wp.pi * SMOOTHING_LENGTH**3))
VIS_MU = wp.constant(0.0 * RHO_0)
