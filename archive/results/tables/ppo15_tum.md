# ppo15 — tum

**Headline:** `finished` = trajectories with no tracking failure in any repeat (the paper's criterion, averaged over seeds); `ate_common_m` = mean ATE on the 7 trajectories that EVERY method and seed finishes (rgbd_dataset_freiburg1_desk, rgbd_dataset_freiburg1_desk2, rgbd_dataset_freiburg1_plant, rgbd_dataset_freiburg1_room, rgbd_dataset_freiburg1_rpy, rgbd_dataset_freiburg1_teddy, rgbd_dataset_freiburg1_xyz), so accuracy is compared on identical, fully tracked sequences.

ATE: first sub-trajectory before a tracking failure (authors' metric), mean over held-out trajectories x repeats. `d_*` columns are relative to the baseline.

**Read ATE together with `ate_coverage`:** a policy that fails early gets a short first segment and a small, flattering ATE. Only compare ATE between runs with similar coverage.
`ate_all_m` is coverage-aware: every tracked segment between failures is aligned separately (>= 10 poses) and errors are pooled (length-weighted RMSE); `ate_all_cov` is the share it covers.

| variant   |   seeds |   finished |   finished_of |   ate_common_m |   ate_common_sd |   ate_m |   ate_median_m |   ate_coverage |   ate_all_m |   ate_all_cov |   tracked |   failures_per_traj |   failures_sd |   keyframe_rate |   train_reward |   train_valid |   train_fail_per_1k |       steps |   minutes |
|:----------|--------:|-----------:|--------------:|---------------:|----------------:|--------:|---------------:|---------------:|------------:|--------------:|----------:|--------------------:|--------------:|----------------:|---------------:|--------------:|--------------------:|------------:|----------:|
| s0        |       1 |      7.000 |             9 |          0.526 |             nan |   0.488 |          0.534 |          0.954 |       0.484 |         0.989 |     0.989 |               0.259 |           nan |           0.191 |          0.001 |         0.917 |               1.992 | 1500000.000 |    49.147 |
