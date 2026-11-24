"""Exact action-alignment check: replay single steps of a rollout CSV in Python MuJoCo 3.17.0.

python Tools/replay_check.py rollout_smooth.csv     (needs the mujoco==3.17.0 package)
For sampled rows t: set (q, qd) = state_t, ctrl = a_t (or a_{t-1}, a_{t+1}), one mj_step, compare with next_state_t.
The recorded action is aligned if it reproduces next_state to float32 rounding and the shifted ones do not.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
name = sys.argv[1] if len(sys.argv) > 1 else "rollout.csv"
rows = np.loadtxt(PROJECT / "Assets/CA3/Data" / name, delimiter=",", dtype=np.float32)
model = mujoco.MjModel.from_xml_path(str(PROJECT / "Robots/ufactory_lite6/lite6.xml"))
model.opt.timestep = 0.002
data = mujoco.MjData(model)
a = rows[:, 12:18]
idx = np.arange(1, len(rows) - 1)
changed = idx[np.any(a[idx] != a[idx - 1], 1) | np.any(a[idx + 1] != a[idx], 1)]  # rows where the shift matters
idx = np.random.default_rng(0).choice(changed, min(3000, len(changed)), replace=False)
print(f"{name}: {len(idx)} sampled rows where a_t differs from a neighbour")
for label, shift in [("a_t (recorded)", 0), ("a_{t-1}", -1), ("a_{t+1}", +1)]:
    err = []
    for t in idx:
        mujoco.mj_resetData(model, data)
        data.qpos[:6], data.qvel[:6] = rows[t, :6], rows[t, 6:12]
        data.ctrl[:] = a[t + shift]
        mujoco.mj_step(model, data)
        err.append(np.abs(np.r_[data.qpos[:6], data.qvel[:6]] - rows[t, 18:30]).max())
    err = np.array(err)
    print(f"  {label:15s} max|state' error| median {np.median(err):.1e}, 99% {np.quantile(err, .99):.1e}, max {err.max():.1e}")
