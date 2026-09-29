# Archive: superseded results (kept for history, do not use for comparisons)

| Path | What | Why superseded |
|---|---|---|
| `results/eval/tartanair/pilot`, `results/tables/pilot_*`, `results/figures/pilot_*` | first pipeline test: all 12 setups, 20k steps | no obs-normalization warm-up (normalization never activated); old 3-trajectory TartanAir test set; buggy GT re-initialization (inflated failure counts) |
| `results/eval/tartanair/pilot2*`, `results/tables/pilot2_*`, `results/figures/pilot2_*` | same with warm-up and 6 test trajectories | still the buggy GT re-initialization (failure counts ~10x inflated); 20k steps only |
| `results/eval/tum/tum_check`, `tum_check_policy` | first TUM check on 3 sequences | untuned SVO settings + buggy re-initialization |
| `results/eval/tum/tum_full` | SVO rules on all 9 TUM sequences | untuned SVO settings + buggy re-initialization. Current reference: `results/eval/tum/svo_rules_std` |
| `scripts/tum_partial_check.sh` | one-off helper to evaluate TUM while it was still downloading | no longer needed |

The matching training runs are in `runs/archive/` (not in git). Details of the bugs and fixes: `docs/modifications.md`.
Chronology: `docs/log.md`.
