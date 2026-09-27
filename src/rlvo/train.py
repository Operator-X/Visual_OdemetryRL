"""Training: the authors' PPO (masked rollout buffer, privileged critic, Perceiver policy) + our env and logging."""
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_linear_fn

import rlvo
from policies.attention_policy import CustomActorCriticPolicy
from rl_algorithms.ppo import PPO

from rlvo.env import EnvConfig, RewardConfig, RLVOEnv
from rlvo.evaluate import evaluate


def load_config(*paths, overrides=()):
    """Merge YAML configs left to right (base first), then dotlist overrides like 'agent.gamma=0.9'."""
    cfg = OmegaConf.merge(*[OmegaConf.load(p) for p in paths], OmegaConf.from_dotlist(list(overrides)))
    return cfg


def env_config(cfg):
    e = OmegaConf.to_container(cfg.env, resolve=True)
    return EnvConfig(reward=RewardConfig(**e.pop('reward', {})), **e)


def make_envs(cfg, val=True):
    ecfg = env_config(cfg)
    data = str(rlvo.ROOT / cfg.data.tartan_dir)
    params = str(rlvo.SVO_PARAMS / cfg.data.svo_params)
    calib = str(rlvo.ROOT / cfg.data.calib)
    env = RLVOEnv(params, calib, data, cfg.n_envs, 'train', ecfg, initialize_glog=True, seed=cfg.seed)
    val_env = None
    if val:
        from dataloader.tartan_loader import test_split
        n_val = sum(1 for t in Path(data).glob("*/*/P*") if any(s in str(t) for s in test_split))
        val_env = RLVOEnv(params, calib, data, n_val, 'val', ecfg, initialize_glog=False, seed=cfg.seed)
    return env, val_env


def make_policy_kwargs(env):
    return dict(
        encoder_kwargs=dict(variable_feature_dim=3, obs_dim_variable=180 * 3, obs_dim_fixed=env.policy_obs_dim_fixed,
                            critique_dim=env.critique_dim),
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        log_std_init=-0.0,
    )


class CSVRolloutLogger(BaseCallback):
    """Per-rollout training metrics to CSV (the reference only logs to wandb)."""

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.t0 = time.time()
        self._fail = 0
        self._steps = 0
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(["iteration", "timesteps", "wall_s", "reward_per_step", "valid_ratio", "keyframe_rate",
                          "failures_per_1k", "position_reward", "rotation_reward"])
        self._pr = self._rr = 0.0

    def _on_step(self):
        for info in self.locals["infos"]:
            self._fail += int(info.get("svo_failure", False))
            self._pr += info.get("position_reward", 0.0)
            self._rr += info.get("rotation_reward", 0.0)
        self._steps += len(self.locals["infos"])
        return True

    def _on_rollout_end(self):
        buf = self.locals["rollout_buffer"]
        valid = buf.valid_mask
        nv = max(valid.sum(), 1)
        self._w.writerow([self.model.iteration if hasattr(self.model, "iteration") else -1, self.model.num_timesteps,
                          round(time.time() - self.t0, 1), float(buf.rewards.mean()), float(valid.mean()),
                          float((buf.actions[:, :, 0] * valid).sum() / nv), 1000.0 * self._fail / max(self._steps, 1),
                          self._pr / max(self._steps, 1), self._rr / max(self._steps, 1)])
        self._f.flush()
        self._fail = self._steps = 0
        self._pr = self._rr = 0.0


class RLVOPPO(PPO):
    """Reference PPO; evaluation writes our metrics to CSV instead of wandb (checkpointing stays in learn())."""

    def evaluation_epoch(self, val_env):
        self.policy.set_training_mode(False)
        res = evaluate(val_env, self.policy, device=self.device)
        path = os.path.join(self.log_dir, "eval.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["iteration", "timesteps"] + list(res[0].keys()))
            if new:
                w.writeheader()
            for r in res:
                w.writerow({"iteration": self.iteration, "timesteps": self.num_timesteps, **r})
        ates = [r["ate"] for r in res]
        print(f"[eval] iter {self.iteration}: ATE {np.round(ates, 3)}  tracked {[round(r['tracked_frac'], 2) for r in res]}")


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(rlvo.ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def train(cfg, run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    (run_dir / "meta.json").write_text(json.dumps({"git_commit": git_commit(), "start": time.ctime()}, indent=2))

    torch.set_num_threads(cfg.torch_threads)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env, val_env = make_envs(cfg)
    env.seed(cfg.seed) if hasattr(env, "seed") else None
    a = cfg.agent
    n_steps = int(a.n_steps)
    batch = int(a.batch_size) if a.batch_size > 0 else n_steps * cfg.n_envs   # <=0 -> full batch (as the authors)
    model = RLVOPPO(
        tensorboard_log=None, log_dir=str(run_dir), policy=CustomActorCriticPolicy,
        policy_kwargs=make_policy_kwargs(env), env=env,
        n_epochs=a.n_epochs, gae_lambda=a.gae_lambda, gamma=a.gamma, n_steps=n_steps, ent_coef=a.ent_coef,
        vf_coef=a.vf_coef, max_grad_norm=a.max_grad_norm, batch_size=batch,
        learning_rate=get_linear_fn(a.lr_start, a.lr_end, 1.0), clip_range=0.2, use_sde=False, verbose=0,
        seed=cfg.seed, wandb_logging=False, wandb_tag=None, wandb_group=None, config=cfg, device="cpu",
    )
    t0 = time.time()
    model.learn(total_timesteps=int(cfg.total_timesteps), log_interval=None, eval_interval=cfg.val_interval,
                val_env=val_env, callback=CSVRolloutLogger(str(run_dir / "train.csv")))
    # final checkpoint + evaluation
    pol = run_dir / "Policy"
    pol.mkdir(exist_ok=True)
    model.policy.save(str(pol / "final.pth"))
    env.save_rms(str(pol / "final_rms.npz"))
    model.evaluation_epoch(val_env)
    meta = json.loads((run_dir / "meta.json").read_text())
    meta.update(end=time.ctime(), train_hours=(time.time() - t0) / 3600, timesteps=int(model.num_timesteps))
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return model
