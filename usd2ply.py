import os
import tarfile
import tempfile

import numpy as np
import zstandard as zstd
from joblib import Parallel, cpu_count, delayed
from plyfile import PlyData, PlyElement
from pxr import Usd, UsdGeom


def usd_to_ply_frame(stage, output_ply_path, frame_time):
    vertices = []

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.PointInstancer):  # type: ignore
            point_instancer = UsdGeom.PointInstancer(prim)  # type: ignore
            positions_attr = point_instancer.GetPositionsAttr()
            if positions_attr:
                positions = positions_attr.Get(frame_time)
                if positions:
                    for pos in positions:
                        vertices.append([pos[0], pos[1], pos[2], 0])  # obj_id设为0
            break

    data = np.array(
        [tuple(vertex) for vertex in vertices],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("obj_id", "u1")],
    )
    el = PlyElement.describe(data, "vertex")
    PlyData([el]).write(output_ply_path)


def process_single_frame(usd_file_path, temp_dir, frame_index, frame_time):
    stage = Usd.Stage.Open(usd_file_path)  # type: ignore
    if not stage:
        print(f"Error: Cannot open USD file {usd_file_path} in thread")
        return

    ply_file = os.path.join(temp_dir, f"frame_{frame_index:04d}.ply")
    usd_to_ply_frame(stage, ply_file, frame_time)
    return ply_file


def usd_to_ply(usd_file_path):
    stage = Usd.Stage.Open(usd_file_path)  # type: ignore
    if not stage:
        print(f"Error: Cannot open USD file {usd_file_path}")
        return

    start_time = stage.GetStartTimeCode()
    end_time = stage.GetEndTimeCode()

    base_name = os.path.splitext(os.path.basename(usd_file_path))[0]
    archive_file_path = os.path.join(
        os.path.dirname(usd_file_path), f"{base_name}.tar.zst"
    )

    n_jobs = max(1, int(cpu_count() * 0.9))

    with tempfile.TemporaryDirectory() as temp_dir:
        if start_time == end_time:
            print("Static scene detected, exporting single frame")
            ply_file = os.path.join(temp_dir, f"{base_name}_0000.ply")
            usd_to_ply_frame(stage, ply_file, start_time)
        else:
            frame_count = int(end_time - start_time) + 1
            print(
                f"Animation detected, exporting {frame_count} frames using {n_jobs} parallel threads"
            )
            frame_args = [
                (usd_file_path, temp_dir, frame_index, start_time + frame_index)
                for frame_index in range(frame_count)
            ]
            Parallel(n_jobs, verbose=1)(
                delayed(process_single_frame)(usd_path, temp_dir, frame_idx, frame_time)
                for usd_path, temp_dir, frame_idx, frame_time in frame_args
            )

        print(f"Creating zstd compressed archive: {archive_file_path}")
        cctx = zstd.ZstdCompressor(level=3, threads=n_jobs)

        with open(archive_file_path, "wb") as f:
            with cctx.stream_writer(f) as compressor:
                with tarfile.open(fileobj=compressor, mode="w|") as tar:
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, temp_dir)
                            tar.add(file_path, arcname=arcname)

    print("Export completed!")


if __name__ == "__main__":
    usd_file_path = "/home/zy/mzh/WarpSPH/example_sph.usd"
    usd_to_ply(usd_file_path)
