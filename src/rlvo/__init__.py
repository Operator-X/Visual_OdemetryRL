"""RL-VO: replication and extensions of "Reinforcement Learning Meets Visual Odometry" (ECCV 2024).

Importing this package makes the compiled svo_env module and the read-only reference code importable.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "reference" / "rl_vo"
SVO_BUILD = ROOT / "third_party" / "svo-lib" / "build" / "svo_env"
SVO_PARAMS = ROOT / "third_party" / "svo-lib" / "svo_env" / "param"
DATA = ROOT / "data"

for p in (SVO_BUILD, REFERENCE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# The reference code predates NumPy 2 (np.Inf was removed). Keep reference/ untouched.
if not hasattr(np, "Inf"):
    np.Inf = np.inf
