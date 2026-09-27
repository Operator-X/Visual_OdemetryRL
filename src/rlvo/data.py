"""Dataloaders: TartanAir with K future poses (for a longer-horizon privileged critic) + image augmentation."""
import os

import numpy as np
import torch

import rlvo  # noqa: F401  (sets up sys.path for svo_env and reference/)
import svo_env
from dataloader.tartan_loader import TartanLoader


class TartanLoaderK(TartanLoader):
    """Same as the reference TartanLoader, but returns poses for t, t+1, ..., t+K (shape [n, K+1, 7]).

    With K=1 the output is identical to the reference loader ([n, 2, 7]).
    """

    def __init__(self, root_path, mode, num_envs, val_traj_ids=None, traj_name=None, n_future=1):
        self.n_future = n_future
        super().__init__(root_path, mode, num_envs, val_traj_ids, traj_name)

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

    def resample(self, mask):
        k = int(mask.sum())
        if k:
            self.gain[mask] = self.rng.uniform(*self.gain_rng, k)
            self.gamma[mask] = self.rng.uniform(*self.gamma_rng, k)
            self.noise[mask] = self.rng.uniform(*self.noise_rng, k)

    def __call__(self, images, new_seq):
        self.resample(np.asarray(new_seq, dtype=bool))
        out = np.empty_like(images)
        x = np.arange(256, dtype=np.float32) / 255.0
        for i in range(self.n):
            g = self.gain[i] * (1.0 + self.rng.uniform(-self.jitter, self.jitter))
            lut = np.clip(255.0 * g * np.power(x, self.gamma[i]), 0, 255)
            img = lut[images[i, :, :, 0]]
            if self.noise[i] > 0:
                img = img + self.rng.normal(0.0, self.noise[i], img.shape).astype(np.float32)
            out[i, :, :, 0] = np.clip(img, 0, 255).astype(np.uint8)
        return out
