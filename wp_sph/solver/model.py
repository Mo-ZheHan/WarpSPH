import multiprocessing
import os
from typing import cast

import numpy as np
import trimesh
from joblib import Memory, Parallel, delayed

from ..main import *

cache_dir = os.path.join(os.path.dirname(__file__), ".model_cache")
memory = Memory(cache_dir, verbose=0)


@memory.cache
def _sample_points(filename, scale, spacing):
    mesh = trimesh.load(model_dir(filename), force="mesh")
    mesh = cast(trimesh.Trimesh, mesh)
    mesh.vertices *= scale
    bounds = mesh.bounds

    # Sample points
    x = np.arange(bounds[0][0], bounds[1][0], spacing)
    y = np.arange(bounds[0][1], bounds[1][1], spacing)
    z = np.arange(bounds[0][2], bounds[1][2], spacing)
    grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), -1).reshape(-1, 3)

    if grid.shape[0] == 0:
        return grid

    n_jobs = max(1, int(0.9 * multiprocessing.cpu_count()))
    n_tasks = n_jobs * 8
    print(f"Using {n_jobs} parallel jobs and {n_tasks} tasks for model sampling.")

    def check_contains_segment(points_segment, mesh_object):
        return mesh_object.contains(points_segment)

    grid_segments = np.array_split(grid, n_tasks)
    grid_segments = [segment for segment in grid_segments if segment.shape[0] > 0]

    if not grid_segments:
        inside = np.array([], dtype=bool)
    else:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(check_contains_segment)(segment, mesh) for segment in grid_segments
        )
        inside = np.concatenate(results)  # type: ignore

    return grid[inside]


def load_model(
    filename,
    scale=1.0,
    pos=np.zeros(3),
    rot=np.eye(3),
    spacing=DIAMETER,
):
    points = _sample_points(filename, float(scale), float(spacing))
    points @= np.array(rot)
    return points + np.array(pos)
