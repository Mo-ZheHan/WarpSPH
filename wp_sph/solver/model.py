from typing import Optional, cast

import numpy as np
import trimesh
from warp.sim import Mesh
from warp.sim.model import ModelBuilder, Transform

from ..main import *


class RigidBuilder:
    def __init__(self):
        self.builder = ModelBuilder()

    def add_rigid(
        self,
        obj_path,
        scale=1.0,
        origin=None,
        spacing=DIAMETER,
        density=RHO_0,
        is_visible=True,
    ):
        mesh = trimesh.load(obj_path, force="mesh")
        mesh = cast(trimesh.Trimesh, mesh)
        mesh.vertices *= scale
        bounds = mesh.bounds

        # Sample points
        x = np.arange(bounds[0][0], bounds[1][0], spacing)
        y = np.arange(bounds[0][1], bounds[1][1], spacing)
        z = np.arange(bounds[0][2], bounds[1][2], spacing)
        grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), -1).reshape(-1, 3)
        inside = mesh.contains(grid)

        self.builder.add_shape_mesh(
            body=self.builder.add_body(origin=origin),
            mesh=Mesh(mesh.vertices, mesh.faces),  # type: ignore
            density=density,
            is_visible=is_visible,
        )

        return grid[inside]

    def finalize(self):
        return self.builder.finalize()
