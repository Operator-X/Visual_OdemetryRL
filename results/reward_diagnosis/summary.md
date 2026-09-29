# Reward diagnosis

## A. Strategy level (TUM, 9 seqs, tum_tuned, GT init)

ATE on the 0 sequences every strategy finishes: .

| strategy   |   pos_reward |   norm_reward |   kf_penalty |   frac_positive |   mean_err_m |   failures |   tracked |   total_reward |   ATE_common(0) |   ATE_all |
|:-----------|-------------:|--------------:|-------------:|----------------:|-------------:|-----------:|----------:|---------------:|----------------:|----------:|
| SVO rules  |      0.00184 |       8e-05   |       0      |         0.99725 |      0.01685 |          3 |   0.98857 |        0.00184 |             nan |   0.45184 |
| p_kf=0.05  |      0.00151 |      -0.00132 |      -1e-05  |         0.93255 |      0.04973 |        119 |   0.77118 |        0.00151 |             nan |   0.05812 |
| p_kf=0.1   |      0.00162 |      -0.00085 |      -1e-05  |         0.95621 |      0.03903 |         90 |   0.82767 |        0.00161 |             nan |   0.07199 |
| p_kf=0.2   |      0.00178 |      -0.0003  |      -2e-05  |         0.98854 |      0.02321 |         26 |   0.93836 |        0.00176 |             nan |   0.26766 |
| p_kf=0.3   |      0.0018  |      -0.00013 |      -3e-05  |         0.99014 |      0.02092 |         18 |   0.95281 |        0.00177 |             nan |   0.2597  |
| p_kf=0.5   |      0.00183 |       8e-05   |      -5e-05  |         0.99575 |      0.01741 |          8 |   0.98082 |        0.00178 |             nan |   0.42286 |
| p_kf=1.0   |      0.00184 |       0.00016 |      -0.0001 |         0.99716 |      0.01642 |          2 |   0.99087 |        0.00174 |             nan |   0.4951  |


Spearman rank correlation between reward and accuracy (higher = reward ranks strategies like true accuracy does; 1 = perfect):

- total_reward: +nan
- pos_reward: +nan
- norm_reward: +nan


## B. Decision level (TartanAir train, random keyframes p=0.3)

Position reward k frames after a keyframe vs no keyframe at t. Effect size d = diff / pooled std (|d| < 0.2 is conventionally 'negligible').

| reward             |   k |   mean_after_kf |   mean_after_no_kf |      diff |   effect_size_d |   n_kf |   n_no_kf |
|:-------------------|----:|----------------:|-------------------:|----------:|----------------:|-------:|----------:|
| absolute (authors) |   1 |        0.001259 |           0.001289 | -3.1e-05  |       -0.021341 |   1731 |      3995 |
| absolute (authors) |   2 |        0.001269 |           0.001168 |  0.0001   |        0.061538 |   1749 |      4018 |
| absolute (authors) |   3 |        0.001241 |           0.001062 |  0.000179 |        0.097607 |   1744 |      4005 |
| absolute (authors) |   4 |        0.001241 |           0.000947 |  0.000294 |        0.148693 |   1738 |      3994 |
| absolute (authors) |   5 |        0.001191 |           0.000856 |  0.000335 |        0.155499 |   1731 |      3984 |
| normalized         |   1 |        0.001114 |           0.001145 | -3.1e-05  |       -0.016488 |   1731 |      3995 |
| normalized         |   2 |        0.001116 |           0.001018 |  9.8e-05  |        0.047707 |   1749 |      4018 |
| normalized         |   3 |        0.001093 |           0.000896 |  0.000197 |        0.088913 |   1744 |      4005 |
| normalized         |   4 |        0.001088 |           0.000778 |  0.00031  |        0.132218 |   1738 |      3994 |
| normalized         |   5 |        0.00107  |           0.000668 |  0.000401 |        0.162631 |   1731 |      3984 |

## Interpretation (2026-09-29)
- The strategy-level ranking by total reward matches the ATE ranking among robust strategies (rules > p0.5 > p0.3 > p1.0;
  only plant and xyz are finished by all four, so it's indicative). But rewards differ by ~5% while ATE differs up to 2x:
  99.7% of steps get a positive position reward on TUM (1.7 cm error vs the 0.2 m threshold).
- Local vs global: always-keyframe has the LOWEST 5-frame-window error (1.6 cm) but the WORST ATE (0.309 m). The
  window reward sees local consistency, not long-term drift. Only the keyframe penalty ranks that strategy lower.
- Decision level: a keyframe's effect on the position reward is negligible and delayed (d = -0.02 at k=1 rising to
  +0.16 at k=5). At gamma=0.6 the k=4-5 benefit is weighted 0.08-0.13, so the agent mostly sees the immediate cost.
  This explains why the PPO run kept reducing keyframes.
- The normalized-error reward gives the same effect sizes: not a fix.
- Metric: `ate_all` is also gamed with many failures (p=0.05: 119 failures, ate_all 0.058 m). Use finished counts +
  ATE on commonly finished sequences as the headline.
- Priorities: gamma 0.9/0.99, a longer reward window (traj_length ~20), failure penalty. normalized_error deprioritized.
