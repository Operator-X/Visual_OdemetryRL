"""Dataloaders: TartanAir with K future poses (for a longer-horizon privileged critic) + image augmentation."""
import os

import cv2
import numpy as np
import torch

import rlvo  # noqa: F401  (sets up sys.path for svo_env and reference/)
import svo_env
from dataloader.tartan_loader import TartanLoader, test_split


def _matches(traj, patterns):
    return any(p in traj for p in patterns)


def split_trajectories(data_dir, extra_val=(), val_include=None):
    """(train, val) trajectory dirs.

    Held out (never trained on) = the authors' (DPVO) test split + our extra held-out trajectories.
    val = the held-out ones that also match `val_include` (None = all held out). Held-out trajectories outside
    val_include are simply unused, so the evaluation set stays fixed when more data is downloaded.
    """
    keys = list(test_split) + list(extra_val)
    trajs = sorted(str(t) for t in __import__("pathlib").Path(data_dir).glob("*/*/P*"))
    held_out = [t for t in trajs if _matches(t, keys)]
    val = [t for t in held_out if val_include is None or _matches(t, val_include)]
    return [t for t in trajs if t not in held_out], val


class TartanLoaderK(TartanLoader):
    """Same as the reference TartanLoader, but returns poses for t, t+1, ..., t+K (shape [n, K+1, 7]).

    With K=1 the output is identical to the reference loader ([n, 2, 7]).
    """

    def __init__(self, root_path, mode, num_envs, val_traj_ids=None, traj_name=None, n_future=1, extra_val=(),
                 val_include=None):
        self.n_future = n_future
        self.extra_val = list(extra_val)
        self.val_include = None if val_include is None else list(val_include)
        super().__init__(root_path, mode, num_envs, val_traj_ids, traj_name)

    def is_test_scene(self, scene):
        return super().is_test_scene(scene) or any(x in scene for x in self.extra_val)

    def extract_trajectories(self):
        # Reference logic + val_include filter (the reference asserts before we could filter, so reimplemented).
        train, val = split_trajectories(self.root_path, self.extra_val, self.val_include)
        if self.mode == 'train':
            self.trajectories_paths = train
        elif self.mode == 'val':
            self.trajectories_paths = sorted(val)
            if self.val_traj_ids != -1 and self.val_traj_ids is not None:
                self.trajectories_paths = [self.trajectories_paths[i] for i in self.val_traj_ids]
            assert self.num_envs == len(self.trajectories_paths), (self.num_envs, len(self.trajectories_paths))
        else:
            raise ValueError(f'Mode {self.mode} not defined')

    def _pose_key(self, traj_path):
        return '_'.join(traj_path.split(os.sep)[2:])  # same (path-depth dependent) key as the reference

    def __getitem__(self, idx):
        # Mirrors reference/rl_vo/dataloader/tartan_loader.py::__getitem__, generalized to K future poses.
        self.internal_idx += 1
        traj_time_idx = self.internal_idx - self.start_dataloader_idx
        self.select_new_traj = np.logical_or(traj_time_idx >= self.nr_samples_per_traj[self.traj_idx],
                                             self.select_new_traj)
        if self.mode == 'train':
            self.traj_idx[self.select_new_traj] = torch.randint(0, len(self.trajectories_paths),
                                                                [self.select_new_traj.sum()]).numpy()
        self.start_dataloader_idx[self.select_new_traj] = self.internal_idx
        new_sequence_mask = self.select_new_traj.copy()
        self.select_new_traj[:] = False
        traj_time_idx = (self.internal_idx - self.start_dataloader_idx).astype('int')

        image_paths = [os.path.join(self.trajectories_paths[self.traj_idx[i]], 'image_left_gray',
                                    '{:06d}_left.jpg'.format(traj_time_idx[i])) for i in range(self.num_envs)]
        images = svo_env.load_image_batch(image_paths, self.num_envs, self.img_h, self.img_w)

        K = self.n_future
        poses = np.zeros([self.num_envs, K + 1, 7])
        for i in range(self.num_envs):
            last = self.nr_samples_per_traj[self.traj_idx[i]] - 1
            idx = np.minimum(traj_time_idx[i] + np.arange(K + 1), last).astype('int')
            poses[i] = self.poses[self._pose_key(self.trajectories_paths[self.traj_idx[i]])][idx, :]
        return images, poses, new_sequence_mask


class PhotometricAugmenter:
    """Domain randomization on uint8 grayscale batches [n, H, W, 1].

    Per sequence (resampled when an env starts a new sequence): global gain, gamma (camera response), noise level.
    Per frame: small brightness jitter (auto-exposure flicker) and Gaussian noise. Kept mild on purpose, because
    SVO is a direct method and relies on photometric consistency between frames.
    """

    def __init__(self, num_envs, gain=(0.7, 1.3), gamma=(0.7, 1.4), noise_std=(0.0, 5.0), jitter=0.03, seed=0):
        self.n = num_envs
        self.gain_rng, self.gamma_rng, self.noise_rng, self.jitter = gain, gamma, noise_std, jitter
        self.rng = np.random.default_rng(seed)
        self.gain = np.ones(num_envs)
        self.gamma = np.ones(num_envs)
        self.noise = np.zeros(num_envs)
        self.resample(np.ones(num_envs, dtype=bool))
        self._x = np.arange(256, dtype=np.float32) / 255.0
        self._bank = None  # unit-variance noise images, created lazily for the image size (fast vs per-pixel RNG)

    def _noise(self, shape):
        if self._bank is None or self._bank.shape[1:] != shape:
            self._bank = self.rng.standard_normal((16,) + shape, dtype=np.float32)
        return self._bank[self.rng.integers(len(self._bank))]

    def resample(self, mask):
        k = int(mask.sum())
        if k:
            self.gain[mask] = self.rng.uniform(*self.gain_rng, k)
            self.gamma[mask] = self.rng.uniform(*self.gamma_rng, k)
            self.noise[mask] = self.rng.uniform(*self.noise_rng, k)

    def __call__(self, images, new_seq):
        self.resample(np.asarray(new_seq, dtype=bool))
        out = np.empty_like(images)
        for i in range(self.n):
            g = self.gain[i] * (1.0 + self.rng.uniform(-self.jitter, self.jitter))
            lut = np.clip(255.0 * g * np.power(self._x, self.gamma[i]), 0, 255).astype(np.uint8)
            img = cv2.LUT(images[i, :, :, 0], lut)
            if self.noise[i] > 0:
                f = cv2.scaleAdd(self._noise(img.shape), float(self.noise[i]), img.astype(np.float32))
                img = cv2.convertScaleAbs(cv2.max(f, 0.0))   # clamp at 0, then saturating cast to uint8
            out[i, :, :, 0] = img
        return out
