from typing import cast

import numpy as np
import trimesh

from ..main import *


def load_model(
    filename,
    scale=1.0,
    pos=np.zeros(3),
    rot=np.eye(3),
    spacing=DIAMETER,
):
    mesh = trimesh.load(model_dir(filename), force="mesh")
    mesh = cast(trimesh.Trimesh, mesh)
    mesh.vertices *= scale
    mesh.vertices @= rot
    mesh.vertices += pos
    bounds = mesh.bounds

    # Sample points
    x = np.arange(bounds[0][0], bounds[1][0], spacing)
    y = np.arange(bounds[0][1], bounds[1][1], spacing)
    z = np.arange(bounds[0][2], bounds[1][2], spacing)
    grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), -1).reshape(-1, 3)
    inside = mesh.contains(grid)

    return grid[inside]
