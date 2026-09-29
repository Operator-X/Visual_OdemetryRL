# Project log (chronological)

Full history of decisions and findings, moved out of CLAUDE.md on 2026-09-29 to keep CLAUDE.md short. Newest at the
bottom. Superseded results mentioned here were moved to `archive/` (see `archive/README.md`).

- 2026-09-27: Project scaffolded. Paper + reference code added. - 2026-09-27: Environment: plain venv, Python 3.12.
- 2026-09-27: svo-lib builds natively on the M3 Pro. Smoke test passes: 12 envs, initializes then tracks on synthetic images. The Mac is the main training machine.
  Calibration yamls are in `data/calibration/`.
- 2026-09-27: Real TartanAir data runs through the authors' VecSVOEnv (~1000 env-steps/s, 10 envs). Local subset: japanesealley,
  carwelding, westerndesert (Easy).
- 2026-09-27: Implemented all proposed modifications as config switches (docs/modifications.md). Fixed an ARM-only SVO
  crash (warpAffine OOB read). Pipeline test: every variant trained 20k steps + evaluated (results/tables/pilot_*).
  EuRoC not downloaded yet (ETH Research Collection rate-limited us; 3 large zips). The TartanAir held-out set
  (3 trajs) is too noisy for real comparisons.
- 2026-09-27: pilot2 = all variants, 20k steps, obs_rms_warmup_steps=200, 6 val trajs, 1 seed, 1 repeat, both argmax and
  stochastic eval (results/tables/pilot2_*, results/figures/pilot2_*). Still far too short for conclusions. Early signals:
  keyframe_paper (5e-3) collapses to ~no keyframes again (2/2 pilots) -> 34 failures/traj; failure_penalty,
  normalized_error and gamma_0.99 learn MORE keyframes (0.65-0.72), and failure_penalty has the fewest failures
  (6.0 vs 9.7/traj stochastic). Mean ATE is dominated by 2 westerndesert trajectories (~10 m); use per-trajectory/median.
- 2026-09-27: TUM-RGBD download (TUM server ~150 KB/s at first, faster later; `scripts/download_tum.sh`). EuRoC blocked:
  the ETH Research Collection rate-limits our IP (HTTP 429 even for a single browser-like request). `scripts/download_euroc.sh`
  is ready; retry later or download via a browser into data/_zips/euroc/.
- 2026-09-27: **Validation on real data:** our SVO heuristics on TUM (GT init, 3 repeats) match the paper's Table 2 SVO
  row within 5%: 360 0.196 vs 0.186, desk 0.647 vs 0.681, desk2 0.880 vs 0.898 (paper RL-SVO: 0.189 / 0.556 / 0.755).
  TUM uses `tartan_test.yaml` params (as the authors' config_eval.yaml). `scripts/evaluate.py --dataset tum [--data-dir]`.
- 2026-09-27: **Full TUM-RGBD check** (SVO heuristics, 9 seqs x 3 repeats, results/eval/tum/tum_full): on the 5 fully
  tracked seqs we are within -19..+16% of the paper's SVO row (desk -5%, desk2 -2%, teddy -9%, plant -19%, xyz +16% = 9 mm).
  floor fails in both. But our SVO LOSES TRACKING on 360 (1x), room (23x), rpy (15x), where the paper's SVO finishes.
  Likely the paper's per-dataset grid-searched SVO params (we use tartan_test.yaml defaults). Fine for RL-vs-heuristic
  (same SVO settings), but not for matching the paper's absolute robustness.
- Authors' evaluation (reference/rl_vo/evaluation/evaluate_runs.py): ATE = first sub-trajectory; a seq counts as finished
  only without any failure ("-" in the tables = not finished). They also report an "intersection" metric: all methods
  compared on the shortest common first segment (coverage-fair). TODO: add it to our compare.py.
- 2026-09-27: **FAST-9 (ARM) vs FAST-10 (x86) corner detector: tested, NO effect.** x86 builds use FAST-10 (SSE2), the ARM
  build FAST-9 (NEON). But SVO re-scores every candidate with fast_corner_score_10 and keeps a corner per grid cell only
  if its score > threshold. FAST-9-only candidates score <= threshold and can never be selected, so the final features
  are identical: TUM 9 seqs x 3 repeats gave byte-identical results, and speed was the same (within noise, <=3%).
  Runtime switch kept: RLVO_FAST_VARIANT=10 (verified with lldb that it calls fast_corner_detect_10). Default FAST-9.
  So the remaining gap to the paper is NOT the detector. Likely causes: grid-searched params, nondeterminism, library versions.
- 2026-09-27: **Reference bug found: GT re-initialization after a tracking failure is mis-indexed** (svo_wrapper.py
  reset_dones: GT flags/poses passed for all envs, other arrays packed; C++ uses the packed index for all). After a real
  failure SVO re-initializes without/with wrong GT and reports a cascade of spurious failures. TUM (SVO rules, 3 reps):
  46.3 failures/run -> 4.0 with `env.fix_reset_gt_indexing: true` (room 22.7 -> 0.7, rpy 14.7 -> 0.3); first-segment ATE
  and coverage identical. Pilot failure counts and the failure_penalty signal were inflated by this. Also: GT init is
  now never done on frames without GT (TUM floor; previously a fatal quaternion check). Recommendation: enable the fix
  for all our experiments. **User decision 2026-09-27: ON by default** (base.yaml `env.fix_reset_gt_indexing: true`;
  EnvConfig default stays False = reference, so tests/test_env.py still checks exact equivalence). Runs before this
  (pilot, pilot2) used the buggy reset.
- 2026-09-27: **SVO settings search on TUM** (scripts/svo_param_search.py, fix_reset_gt_indexing on, 16 configs over
  kfselect_min_disparity/min_angle/min_dist_metric and quality_min_fts). Only quality_min_fts matters (40 -> 25). The keyframe
  thresholds barely matter (FORWARD criterion is dominated by the #tracked-features bounds 90/180). Saved as
  `svo_env/param/tum_tuned.yaml`: 7/8 paper seqs finished in 3/3 runs, mean
  deviation 8% from the paper's SVO row; room 0.817 (+1%), rpy 0.055 (+3%). 360 still fails once per run. Tuned on the
  test set, like the paper's baseline. RL must be evaluated with the SAME SVO params.
- 2026-09-27: **User decision: tum_tuned is the standard for TUM.** `evaluate.py --dataset tum` now uses tum_tuned.yaml
  (for SVO rules AND RL agents); the authors' untuned file is `--dataset tum_default`. Earlier TUM results
  (results/eval/tum/tum_check*, tum_full) were produced with the untuned file and the buggy re-init.
- 2026-09-28/29: **First real PPO run** (runs/ppo15_s0, baseline + fix, 1 seed, obs warm-up; 400k steps, then resumed
  to 1.5M). TUM vs SVO rules (tum_tuned, 3 reps; results/eval/tum/ppo15*, svo_rules_std):
  400k: 7/9 finished, 2.3 failures/run (rules 3.0), keyframe rate 0.19 (rules 0.29), ATE on commonly finished seqs +8% vs rules.
  1.5M: 6/9 finished, 3.7 failures/run, keyframe rate 0.15, ATE +11%. Training reward/step FLAT at ~0.0011 from ~250k
  on, while the keyframe rate keeps falling (0.49 -> 0.20). Interpretation: with the authors' reward the accuracy
  term barely discriminates (0.2 m threshold saturates), so PPO keeps optimizing the keyframe penalty. Robust with
  half the keyframes (reproduces the paper's efficiency claim), but no accuracy gain. Longer training alone is not the
  fix at our scale. Next: reward variants (normalized_error, failure_penalty, no keyframe penalty) x seeds.
  Note: on resume the linear LR schedule restarts from the new total (LR jumped back up at 402k).
- 2026-09-29: **Reward diagnosis** (scripts/reward_diagnosis.py, results/reward_diagnosis/summary.md, no training).
  The authors' reward ranks strategies correctly but weakly (rewards differ ~5% vs ATE up to 2x; 99.7% of steps
  positive on TUM). The 5-frame window measures local error, not drift: always-keyframe has the lowest window error but
  the worst ATE. A keyframe's benefit is delayed (effect size -0.02 at k=1 -> +0.16 at k=5), and gamma=0.6 weights k=4-5
  at 0.08-0.13 -> explains PPO cutting keyframes. normalized_error doesn't help (same effect sizes). ate_all is also
  gamed with many failures. Priorities: gamma 0.9/0.99, longer reward window (traj_length ~20), failure penalty.
- 2026-09-29: **Reward experiment reward1** (5 setups x 1 seed x 400k, warm-up, fix on; results/tables/reward1_*,
  results/figures/reward1_*). The baseline reproduced the earlier ppo15 400k run EXACTLY (same seed: training is
  deterministic). TUM, ATE on the 5 seqs every method finishes: SVO rules 0.510, baseline 0.551, **long_window 0.523
  (-5% vs baseline)**, failure_penalty 0.548 (5/9 finished), gamma_0.99 0.553 (5/9), gamma_0.9 0.601. TartanAir held-out:
  long_window has the fewest failures (0.5/traj vs baseline 1.17, rules 1.5) and the most finished (3/6).
  gamma 0.9/0.99 learn to keyframe EVERY frame (rate ~1.0): they see the delayed benefit, the 1e-4 penalty can't balance
  it, and always-keyframe drifts more (as in the diagnosis). No setup beats SVO rules on TUM accuracy yet.
  Next: confirm long_window with 3 seeds; try long_window + gamma 0.9 + a larger keyframe penalty / constrained RL.
- 2026-09-29: **3-seed check: long_window's seed-0 advantage did NOT hold.** TUM ATE on the 5 commonly finished seqs:
  SVO rules 0.510; baseline 0.551/0.522/0.544 (mean 0.539 +- 0.015); long_window 0.523/0.585/0.607 (0.572 +- 0.044).
  2 of 3 long_window seeds collapsed to keyframing every frame (rate 0.90, 1.00). TartanAir failures/traj: baseline
  1.02 +- 0.17, long_window 0.85 +- 0.34 (within noise). Single-seed results are unreliable here.
  **Cross-experiment finding:** every reward change fails the same way: the keyframe rate slides to an extreme.
  A large penalty (5e-3) -> ~no keyframes; a reward exposing the delayed benefit (long_window, gamma 0.9/0.99) with the
  small penalty (1e-4) -> keyframe every frame. The one penalty weight decides the extreme. Next: control the keyframe
  budget directly (constrained RL / Lagrangian penalty to a target rate ~0.25-0.3), or a penalty sweep with long_window.
- 2026-09-29: **Folder cleanup.** Superseded results (pilot, pilot2, early TUM checks) moved to `archive/` (tracked) and
  `runs/archive/` (not tracked); run logs to `runs/logs/`, download logs to `data/logs/`. CLAUDE.md shortened; this log
  created.
- 2026-09-30: **Behavior cloning of SVO's rules** (scripts/bc.py, runs/bc_rules_s0): 26k demo decisions from TartanAir
  train (rules keyframe rate 0.35); keyframe accuracy 90.1% vs 63.1% majority baseline; grid 100%. TUM (3 reps): ATE on
  the 5 commonly finished seqs 0.520 (rules 0.510), 6/9 finished, keyframe rate 0.38.
- 2026-09-30: **BC -> PPO fine-tuning** (runs/bc_ppo_s0: init_policy=runs/bc_rules_s0, critic_warmup_iters=5, 400k,
  1 seed). TUM ATE on the 5 commonly finished seqs: **0.467 = -8% vs SVO rules** (0.510), -10% vs BC, -15% vs PPO from
  scratch (0.551); 7/9 finished (+360 in 2/3 runs), 3.3 failures/run. Per seq vs rules: desk -30%, desk2 -19% (both
  better than the paper's RL-SVO), plant +15%, teddy +8%, xyz +79%, room +3%, rpy +7%: the mean is driven by desk/desk2;
  the geometric mean of per-seq ratios is ~+5%. Training: keyframe rate stayed 0.35 -> 0.31 (no collapse), failures
  fell to 1.3/1k (lowest of any run). First result beating the rules on the paper's metric, but 1 seed only.
