"""L4.1 timing check: does train_controller's delayed control step reproduce Unity's data->ctrl sequence?

python Tools/check_ctrl_timing.py L4_unity_controller.csv controller.json
python Tools/check_ctrl_timing.py L41_unity_controller.csv controller_l41.json
Feeds the logged Unity states/goals through train_controller.controller_step (same code as the training
rollout) and compares the action each physics step uses with the logged ctrl, for delays 1, 2, 3.
WorldModelController's first frame runs one frame before the first logged step, with the arm still at rest.
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "Assets/CA3"))
import train_controller as tc  # noqa: E402

log = np.loadtxt(PROJECT / "TestResults" / sys.argv[1], delimiter=",", dtype=np.float32)
c = json.loads((tc.ROOT / sys.argv[2]).read_text())
W = np.array(c["W"], dtype=np.float32).reshape(tc.ACTION, tc.FEAT)
B = np.array(c["b"], dtype=np.float32)
us, ug, uc = log[:, :12], log[:, 12:18], log[:, 18:24]
print(f"{sys.argv[1]} ({len(log)} steps) with {sys.argv[2]}")
for d in (1, 2, 3):
    prev, queue = us[0, :6].copy(), [uc[0].copy() for _ in range(d)]
    prev, _ = tc.controller_step(us[0], ug[0], W, B, prev, queue)  # frame -1 (not logged)
    applied = []
    for t in range(len(log)):
        prev, a = tc.controller_step(us[t], ug[t], W, B, prev, queue)
        applied.append(a)
    err = np.abs(np.array(applied) - uc).max(1)
    print(f"  delay {d}{' (CTRL_DELAY)' if d == tc.CTRL_DELAY else ''}: max |applied - Unity ctrl| = {err.max():.2e}, "
          f"steps with error > 1e-3: {(err > 1e-3).sum()}/{len(log)}")
