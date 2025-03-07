import argparse
import os
import sys

import warp as wp

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import wp_sph as wsph

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
    "--num_frames", type=int, default=20000, help="Total number of frames."
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="Print out additional status messages during execution.",
)

args = parser.parse_known_args()[0]

with wp.ScopedDevice(args.device):
    sph_demo = wsph.solver.IISPH(stage_path=args.stage_path, verbose=args.verbose)

    if wsph.MODE == wsph.Mode.DEBUG:
        sph_demo.renderer.paused = True
        for _ in range(args.num_frames):
            sph_demo.step()
            sph_demo.render()
            if sph_demo.window_closed:
                break
        print(
            f"Done. {sph_demo.penetration_times} potential particle penetrations detected."
        )
    else:
        for _ in range(args.num_frames):
            sph_demo.step()
            sph_demo.render()

    if sph_demo.renderer:
        sph_demo.renderer.save()
