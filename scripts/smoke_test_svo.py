"""Smoke test for the compiled svo_env module: run SVO on synthetic images (no dataset needed).

A camera pans sideways over a random textured plane. Checks that SVOEnv constructs, steps without
crashing, and reports its stage and runtime. Pose accuracy is NOT checked here.

    .venv/bin/python scripts/smoke_test_svo.py [--envs 4] [--steps 60]
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/svo-lib/build/svo_env"))
import svo_env  # noqa: E402

PARAMS = ROOT / "third_party/svo-lib/svo_env/param/tartan_train.yaml"
CALIB = ROOT / "data/calibration/tartan_pinhole.yaml"
H, W = 480, 640
OBS_DIM = 180 * 3 + 24  # matches reference/rl_vo/env/svo_wrapper.py


def make_frames(n_steps, shift_px=4, seed=0):
    rng = np.random.default_rng(seed)
    world = rng.integers(0, 255, (H + 64, W + n_steps * shift_px + 64), dtype=np.uint8)
    world = cv2.GaussianBlur(world, (0, 0), 2.0)
    world = cv2.normalize(world, None, 0, 255, cv2.NORM_MINMAX)
    return [world[32:32 + H, 32 + t * shift_px: 32 + t * shift_px + W].copy() for t in range(n_steps)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()
    n = args.envs

    env = svo_env.SVOEnv(str(PARAMS), str(CALIB), n, True)
    env.reset(np.arange(n).astype(np.float64))
    frames = make_frames(args.steps)

    actions = np.zeros([n, 2], dtype=np.float64)
    use_rl = np.zeros([n], dtype=np.float64)  # 0 = SVO's own heuristics
    use_gt_init = np.zeros([n], dtype=np.float64)
    gt_init_pose = -np.ones([n, 7], dtype=np.float64)

    stage_hist, wall = [], []
    for t, img in enumerate(frames):
        images = np.ascontiguousarray(np.repeat(img[None, :, :, None], n, axis=0))  # (n, H, W, 1)
        stamps = np.full([n], t * 5e7, dtype=np.float64)  # 20 Hz in ns
        poses = np.zeros([n, 16])
        obs = np.zeros([n, OBS_DIM])
        dones = np.zeros([n])
        stages = np.zeros([n])
        runtime = np.zeros([n])
        t0 = time.perf_counter()
        env.step(images, stamps, actions, use_rl, poses, obs, dones, stages, runtime, use_gt_init, gt_init_pose)
        wall.append(time.perf_counter() - t0)
        stage_hist.append(int(stages[0]))

    print(f"OK: {args.steps} steps x {n} envs, no crash")
    print("stages (env 0):", "".join(str(s) for s in stage_hist))
    print(f"wall time per vec-step: {1e3 * np.median(wall):.2f} ms (median)")
    T = poses[0].reshape(4, 4).T  # pose buffer is column-major (Eigen default)
    print("final translation env 0:", np.round(T[:3, 3], 3), "(expect +x for this pan; scale is arbitrary)")
    print("nonzero obs entries env 0:", int(np.count_nonzero(obs[0])))


if __name__ == "__main__":
    main()
