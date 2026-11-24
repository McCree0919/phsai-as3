"""L2 sanity check for Assets/CA3/Data/rollout.csv, loaded with the teacher's train.load_rollout().

python Tools/check_rollout.py                      (prints a short report)
python Tools/check_rollout.py rollout_smooth.csv   (L3.1 smoothed-action data)
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

DT = 0.002
spec = importlib.util.spec_from_file_location("train", Path(__file__).resolve().parents[1] / "Assets/CA3/train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
train.STATE, train.ACTION = 12, 6  # Lite 6 (train.py still has the 14/7 defaults; set in L3)
S, A = train.STATE, train.ACTION

NAME = sys.argv[1] if len(sys.argv) > 1 else "rollout.csv"
data = train.load_rollout(NAME)
s, a, s1 = data[:, :S], data[:, S:S + A], data[:, S + A:]
q, qd, q1, qd1 = s[:, :6], s[:, 6:], s1[:, :6], s1[:, 6:]
ok = True


def check(name, cond, detail):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}: {detail}")


print(f"file={train.ROOT / NAME}")
check("shape", data.shape[1] == 30, f"{data.shape} = state {S} + action {A} + next_state {S}")
check("finite", np.isfinite(data).all(), "no NaN/Inf")

# Rows are one continuous trajectory: next_state[t] is exactly state[t+1].
gap = np.abs(s1[:-1] - s[1:]).max()
check("consecutive rows", gap == 0.0, f"max|next_state[t] - state[t+1]| = {gap:g}")

# One MuJoCo step (implicitfast): q_{t+1} = q_t + dt * qd_{t+1}.
res = np.abs(q1 - q - DT * qd1).max()
check("one 0.002 s physics step per row", res < 1e-5, f"max|q' - q - dt*qd'| = {res:.2e} (float32 rounding)")

# Action alignment: fit dqd = f(a - q, qd) per joint, then compare residuals at goal-switch rows
# using a_t (recorded) vs a_{t-1} / a_{t+1} (off by one step).
sw = np.where(np.any(a[1:] != a[:-1], axis=1))[0] + 1
sw = sw[(sw > 1) & (sw < len(a) - 2)]
dqd = qd1 - qd
rms = {}
for name, shift in [("a_t (recorded)", 0), ("a_{t-1}", -1), ("a_{t+1}", +1)]:
    err = []
    for j in range(6):
        X = np.c_[a[:, j] - q[:, j], qd[:, j], np.ones(len(a))]
        w = np.linalg.lstsq(X, dqd[:, j], rcond=None)[0]
        k = np.r_[sw - 1, sw]  # last row of the old goal and first row of the new one
        Xs = np.c_[a[k + shift, j] - q[k, j], qd[k, j], np.ones(len(k))]
        err.append(dqd[k, j] - Xs @ w)
    rms[name] = float(np.sqrt(np.mean(np.square(err))))
best = min(rms, key=rms.get)
check("action_t drives state_t -> state_t+1", best == "a_t (recorded)",
      f"{len(sw)} goal switches, residual RMS of dqd: " + ", ".join(f"{k}={v:.4f}" for k, v in rms.items()))

moving = np.abs(qd).max(axis=1) > 0.01
check("not mostly static", moving.mean() > 0.5, f"{moving.mean():.0%} of rows have max|qd| > 0.01 rad/s")
print(f"distinct actions={len(np.unique(a, axis=0))}, action changes on {np.mean(np.any(a[1:] != a[:-1], 1)):.1%} of steps, "
      f"|a - q| median / 90% / 99% = " + " / ".join(f"{v:.3f}" for v in np.quantile(np.abs(a - q), [.5, .9, .99])) + " rad")
for j in range(6):
    print(f"  joint{j + 1}: q in [{q[:, j].min():+.3f}, {q[:, j].max():+.3f}]  "
          f"action in [{a[:, j].min():+.3f}, {a[:, j].max():+.3f}]  max|qd|={np.abs(qd[:, j]).max():.2f} rad/s")
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
