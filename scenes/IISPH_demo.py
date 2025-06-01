import argparse
import os
import sys

import warp as wp
from tqdm import tqdm

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import wp_sph as wsph
from usd2ply import usd_to_ply

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--device", type=str, default=None, help="Override the default Warp device."
)
parser.add_argument(
    "--stage_path",
    type=lambda x: None if x == "None" else str(x),
    default="example_sph.usd",
    help="Path to the output USD file.",
)
parser.add_argument(
    "--num_frames", type=int, default=100, help="Total number of frames."
)
parser.add_argument(
    "--preview",
    action="store_true",
    help="Enable the preview window.",
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="Print out additional status messages during execution.",
)
parser.add_argument(
    "--no-compress",
    dest="compress",
    action="store_false",
    default=True,
    help="Disable USD to PLY conversion and keep original USD file.",
)

args = parser.parse_known_args()[0]

with wp.ScopedDevice(args.device):
    sph_demo = wsph.solver.IISPH(
        stage_path=args.stage_path, preview=args.preview, verbose=args.verbose
    )
    pbar = tqdm(
        total=args.num_frames,
        desc="Simulation Progress",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    if sph_demo.previewer:
        sph_demo.previewer.paused = True
        for frame in range(args.num_frames):
            sph_demo.step()
            sph_demo.render()
            pbar.update(1)
            if sph_demo.window_closed:
                break
    else:
        for frame in range(args.num_frames):
            sph_demo.step()
            sph_demo.render()
            pbar.update(1)
    pbar.close()

    if sph_demo.renderer:
        sph_demo.renderer.save()

    if args.compress and args.stage_path and os.path.exists(args.stage_path):
        print("\nConverting to compressed PLY format...")
        try:
            usd_to_ply(args.stage_path)
            os.remove(args.stage_path)
        except Exception as e:
            print(f"Conversion failed: {e}")
