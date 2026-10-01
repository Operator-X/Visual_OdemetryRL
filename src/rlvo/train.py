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

from rlvo.data import split_trajectories
from rlvo.env import EnvConfig, RewardConfig, RLVOEnv
from rlvo.evaluate import evaluate


def load_config(*paths, overrides=()):
    """Merge YAML configs left to right (base first), then dotlist overrides like 'agent.gamma=0.9'."""
    cfg = OmegaConf.merge(*[OmegaConf.load(p) for p in paths], OmegaConf.from_dotlist(list(overrides)))
    return cfg


def env_config(cfg):
    e = OmegaConf.to_container(cfg.env, resolve=True)
    return EnvConfig(reward=RewardConfig(**e.pop('reward', {})), **e)


def set_svo_threads(cfg):
    """SVO reads RLVO_SVO_THREADS when an env is constructed and on every image-batch load."""
    os.environ["RLVO_SVO_THREADS"] = str(cfg.get("svo_threads", 8))


def make_envs(cfg, val=True):
    set_svo_threads(cfg)
    ecfg = env_config(cfg)
    data = str(rlvo.ROOT / cfg.data.tartan_dir)
    params = str(rlvo.SVO_PARAMS / cfg.data.svo_params)
    calib = str(rlvo.ROOT / cfg.data.calib)
    extra_val = list(cfg.data.get("extra_val_trajs", []))
    val_include = cfg.data.get("val_include", None)
    val_include = None if val_include is None else list(val_include)
    train_trajs, val_trajs = split_trajectories(data, extra_val, val_include)
    if cfg.n_envs > len(train_trajs):
        raise SystemExit(f"n_envs={cfg.n_envs} > {len(train_trajs)} training trajectories (TartanLoader needs <=)")
    env = RLVOEnv(params, calib, data, cfg.n_envs, 'train', ecfg, initialize_glog=True, seed=cfg.seed,
                  extra_val=extra_val, val_include=val_include)
    val_env = None
    if val:
        val_env = RLVOEnv(params, calib, data, len(val_trajs), 'val', ecfg, initialize_glog=False, seed=cfg.seed,
                          extra_val=extra_val, val_include=val_include)
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

    def __init__(self, path, append=False, wall_offset=0.0):
        super().__init__()
        self.path = path
        self.t0 = time.time() - wall_offset
        self._fail = 0
        self._steps = 0
        append = append and os.path.exists(path)
        self._f = open(path, "a" if append else "w", newline="")
        self._w = csv.writer(self._f)
        if not append:
            self._w.writerow(["iteration", "timesteps", "wall_s", "reward_per_step", "valid_ratio", "keyframe_rate",
                              "failures_per_1k", "position_reward", "rotation_reward", "actual_keyframe_rate",
                              "override_rate"])
        self._pr = self._rr = 0.0
        self._kf = self._ov = self._val = 0

    def _on_step(self):
        for info in self.locals["infos"]:
            self._kf += int(info.get("keyframe_actual", False))
            self._ov += int(info.get("override", False))
            self._val += int(info.get("valid", False))
            self._fail += int(info.get("svo_failure", False))
            self._pr += info.get("position_reward", 0.0)
            self._rr += info.get("rotation_reward", 0.0)
        self._steps += len(self.locals["infos"])
        return True

    def _on_rollout_end(self):
        buf = self.locals["rollout_buffer"]
        valid = buf.valid_mask
        nv = max(valid.sum(), 1)
        rollout = self.model.n_steps * self.model.n_envs   # continuous across resumes (reference resets iteration)
        self._w.writerow([self.model.num_timesteps // rollout - 1, self.model.num_timesteps,
                          round(time.time() - self.t0, 1), float(buf.rewards.mean()), float(valid.mean()),
                          float((buf.actions[:, :, 0] * valid).sum() / nv), 1000.0 * self._fail / max(self._steps, 1),
                          self._pr / max(self._steps, 1), self._rr / max(self._steps, 1),
                          self._kf / max(self._val, 1), self._ov / max(self._val, 1)])
        self._f.flush()
        self._fail = self._steps = 0
        self._pr = self._rr = 0.0
        self._kf = self._ov = self._val = 0


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
        print(f"[eval] {self.num_timesteps} steps: ATE {np.round([r['ate'] for r in res], 3)} "
              f"ATE_all {np.round([r['ate_all'] for r in res], 3)} tracked {[round(r['tracked_frac'], 2) for r in res]}",
              flush=True)


# ------------------------------------------------------------------ checkpoint / resume
CKPT = "checkpoint.pt"


def _rms_state(rms):
    return dict(mean=np.asarray(rms.mean), var=np.asarray(rms.var), count=float(rms.count))


def save_checkpoint(model, env, run_dir, wall_s):
    """Everything needed to continue training: weights, optimizer, step counter, obs normalization, RNG states."""
    state = dict(
        policy=model.policy.state_dict(), optimizer=model.policy.optimizer.state_dict(),
        num_timesteps=int(model.num_timesteps), wall_s=float(wall_s),
        obs_rms=_rms_state(env.obs_rms), obs_rms_new=_rms_state(env.obs_rms_new),
        rms_shared=env.obs_rms is env.obs_rms_new,
        kf_lambda=float(getattr(env, "kf_lambda", 0.0)), kf_integral=float(getattr(env, "kf_integral", 0.0)),
        rng=dict(numpy=np.random.get_state(), torch=torch.get_rng_state()),
    )
    tmp = Path(run_dir) / (CKPT + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, Path(run_dir) / CKPT)   # atomic: a crash mid-save never corrupts the last checkpoint


def load_checkpoint(model, env, val_env, run_dir):
    state = torch.load(Path(run_dir) / CKPT, map_location="cpu", weights_only=False)
    model.policy.load_state_dict(state["policy"])
    model.policy.optimizer.load_state_dict(state["optimizer"])
    model.num_timesteps = state["num_timesteps"]
    for name in ("obs_rms", "obs_rms_new"):
        rms = getattr(env, name)
        rms.mean, rms.var, rms.count = state[name]["mean"], state[name]["var"], state[name]["count"]
    if state["rms_shared"]:
        env.obs_rms = env.obs_rms_new
    if "kf_lambda" in state:
        env.kf_lambda, env.kf_integral = state["kf_lambda"], state["kf_integral"]
    if val_env is not None:
        val_env.obs_rms = env.obs_rms
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    return state


def warmup_obs_rms(env, val_env, vec_steps):
    """Estimate observation-normalization stats before training (SVO heuristics, GT init), then activate them.

    The reference PPO only activates normalization at PPO iteration 10; before that the policy sees raw values.
    env.normalize_obs() updates obs_rms_new on every train-mode step, so stepping is enough to collect the stats.
    """
    env.reset(use_gt_initialization=True)
    zeros = np.zeros([env.num_envs, env.action_dim], dtype=np.int64)
    for _ in range(int(vec_steps)):
        env.step(zeros, use_RL_actions_bool=False, use_gt_initialization=True)
    env.update_rms()                     # obs_rms <- obs_rms_new (same object from now on, as in the reference)
    if val_env is not None:
        val_env.obs_rms = env.obs_rms
    print(f"[warmup] obs normalization from {int(vec_steps) * env.num_envs} samples", flush=True)


def init_from_policy(model, env, val_env, init_run):
    """Start PPO from another run's policy (e.g. behavior cloning) and its observation-normalization stats."""
    init_run = Path(init_run)
    if not init_run.is_absolute():
        init_run = rlvo.ROOT / init_run
    loaded = CustomActorCriticPolicy.load(str(init_run / "Policy" / "final.pth"), device="cpu")
    model.policy.load_state_dict(loaded.state_dict())
    stats = np.load(init_run / "Policy" / "final_rms.npz")
    env.obs_rms.mean, env.obs_rms.var = stats["mean"].reshape(env.obs_rms.mean.shape), stats["var"].reshape(env.obs_rms.var.shape)
    env.obs_rms.count = 1e4                  # keep updating from these stats instead of starting fresh
    env.obs_rms_new = env.obs_rms            # same object, as after the reference's update_rms()
    if val_env is not None:
        val_env.obs_rms = env.obs_rms
    print(f"[init] policy + obs stats from {init_run}", flush=True)


class CriticWarmupCallback(BaseCallback):
    """Freeze the actor (shared keypoint encoder, policy MLP, action head) for the first `iters` PPO updates, so the
    untrained critic can fit the value of the (e.g. behavior-cloned) policy before the policy starts changing."""

    def __init__(self, iters):
        super().__init__()
        self.iters, self._updates = int(iters), 0

    def _actor_params(self):
        p = self.model.policy
        return (list(p.mlp_extractor.variable_encoder.parameters()) + list(p.mlp_extractor.policy_net.parameters())
                + list(p.action_net.parameters()))

    def _set(self, trainable):
        for q in self._actor_params():
            q.requires_grad_(trainable)

    def _on_training_start(self):
        if self.iters > 0:
            self._set(False)
            print(f"[critic-warmup] actor frozen for {self.iters} PPO updates", flush=True)

    def _on_rollout_start(self):
        if getattr(self.model, "iteration", 0) > 0 and self.iters > 0:
            self._updates += 1
            if self._updates == self.iters:
                self._set(True)
                print("[critic-warmup] actor unfrozen", flush=True)

    def _on_step(self):
        return True


class KeyframeLagrangeCallback(BaseCallback):
    """Constrained RL: after every rollout, set the keyframe penalty (Lagrange multiplier, env.kf_lambda) with a PI
    controller so the actual keyframe rate (on steps where the agent acts) tracks `target`. Two-sided: lambda can go
    negative (a keyframe bonus) if the rate falls below target. PI instead of plain gradient ascent on lambda avoids the
    classic Lagrangian oscillation (Stooke et al. 2020, PID Lagrangian). Writes lagrange.csv."""

    def __init__(self, env, target, kp, ki, path):
        super().__init__()
        self.env_ref, self.target, self.kp, self.ki = env, float(target), float(kp), float(ki)
        self.lam0 = float(env.cfg.reward.keyframe_reward)
        self._kf = self._val = 0
        new = not os.path.exists(path)
        self._f = open(path, "a", newline="")
        self._w = csv.writer(self._f)
        if new:
            self._w.writerow(["timesteps", "keyframe_rate", "error", "integral", "lambda"])

    def _on_step(self):
        for info in self.locals["infos"]:
            self._kf += int(info.get("keyframe_actual", False))
            self._val += int(info.get("valid", False))
        return True

    def _on_rollout_end(self):
        rate = self._kf / max(self._val, 1)
        err = rate - self.target
        e = self.env_ref
        e.kf_integral += err
        e.kf_lambda = self.lam0 + self.kp * err + self.ki * e.kf_integral
        self._w.writerow([self.model.num_timesteps, round(rate, 4), round(err, 4), round(e.kf_integral, 4),
                          f"{e.kf_lambda:.6g}"])
        self._f.flush()
        self._kf = self._val = 0


class CheckpointCallback(BaseCallback):
    """Saves a resumable checkpoint every `every` PPO updates (at rollout start = right after an update)."""

    def __init__(self, env, run_dir, every, wall_offset=0.0):
        super().__init__()
        self.env_ref, self.run_dir, self.every = env, run_dir, max(1, int(every))
        self.t0 = time.time() - wall_offset
        self._updates = 0

    def _on_rollout_start(self):
        if getattr(self.model, "iteration", 0) > 0:
            self._updates += 1
            if self._updates % self.every == 0:
                save_checkpoint(self.model, self.env_ref, self.run_dir, time.time() - self.t0)

    def _on_step(self):
        return True


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(rlvo.ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def train(cfg, run_dir, resume=False):
    """Train from scratch, or with resume=True continue from run_dir/checkpoint.pt (config taken from run_dir)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    resuming = resume and (run_dir / CKPT).exists()
    if resuming:
        cfg = OmegaConf.load(run_dir / "config.yaml")
    else:
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
    wall0 = 0.0
    if resuming:
        state = load_checkpoint(model, env, val_env, run_dir)
        wall0 = state["wall_s"]
        env.seed(cfg.seed + state["num_timesteps"])   # fresh SVO randomness after resume
        print(f"[resume] from {state['num_timesteps']} steps", flush=True)
    elif cfg.env.get("residual", False) and cfg.get("residual_init_follow_logit", 0) and not cfg.get("init_policy", None):
        with torch.no_grad():                    # start close to SVO's rules: bias the "follow rule" logit
            model.policy.action_net.bias[0] += float(cfg.residual_init_follow_logit)
        print(f"[residual] follow-rule logit +{cfg.residual_init_follow_logit}", flush=True)
        if cfg.get("obs_rms_warmup_steps", 0) > 0:
            warmup_obs_rms(env, val_env, cfg.obs_rms_warmup_steps)
    elif cfg.get("init_policy", None):
        init_from_policy(model, env, val_env, cfg.init_policy)
    elif cfg.get("obs_rms_warmup_steps", 0) > 0:
        warmup_obs_rms(env, val_env, cfg.obs_rms_warmup_steps)
    remaining = int(cfg.total_timesteps) - int(model.num_timesteps)
    t0 = time.time()
    if remaining > 0:
        callbacks = [CSVRolloutLogger(str(run_dir / "train.csv"), append=resuming, wall_offset=wall0),
                     CheckpointCallback(env, run_dir, cfg.get("checkpoint_every", 10), wall_offset=wall0)]
        if not resuming and cfg.get("critic_warmup_iters", 0) > 0:
            callbacks.append(CriticWarmupCallback(cfg.critic_warmup_iters))
        if cfg.env.get("kf_target", None) is not None:
            lg = cfg.get("kf_lagrange", {})
            callbacks.append(KeyframeLagrangeCallback(env, cfg.env.kf_target, lg.get("kp", 2e-3), lg.get("ki", 5e-3),
                                                      str(run_dir / "lagrange.csv")))
        model.learn(total_timesteps=remaining, log_interval=None, eval_interval=cfg.val_interval,
                    val_env=val_env, callback=callbacks, reset_num_timesteps=not resuming)
    save_checkpoint(model, env, run_dir, wall0 + time.time() - t0)
    # final policy + evaluation
    pol = run_dir / "Policy"
    pol.mkdir(exist_ok=True)
    model.policy.save(str(pol / "final.pth"))
    env.save_rms(str(pol / "final_rms.npz"))
    model.evaluation_epoch(val_env)
    meta = json.loads((run_dir / "meta.json").read_text())
    meta.update(end=time.ctime(), train_hours=(wall0 + time.time() - t0) / 3600, timesteps=int(model.num_timesteps),
                resumed=bool(resuming) or meta.get("resumed", False))
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return model
