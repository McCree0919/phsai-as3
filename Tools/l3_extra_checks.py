"""L3 extra acceptance checks (no retraining, rollout.csv untouched).

python Tools/l3_extra_checks.py   (after Tools/smooth_rollouts.py)
  -> TestResults/L3_extra_checks.txt, L3_smooth_rollouts.png
0) the saved dynamics.pt is the best-val (epoch 5) checkpoint
1) action sensitivity: correct vs shuffled actions (one-step and multi-step at goal switches)
2) L4-style smoothed action episodes: MuJoCo vs world-model recursion
3) ONNX / normalization contract for Unity inference
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import torch

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "TestResults"
spec = importlib.util.spec_from_file_location("train", PROJECT / "Assets/CA3/train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
S, A, ROOT = train.STATE, train.ACTION, train.ROOT

data = train.load_rollout()
split = int(len(data) * (1 - train.VAL_FRACTION))
nz = np.load(ROOT / "norm.npz")
stats = (nz["x_mean"], nz["x_std"], nz["y_mean"], nz["y_std"])
model = train.build_model()
model.load_state_dict(torch.load(ROOT / "dynamics.pt", weights_only=True))
model.eval()
lines = []


def out(text=""):
    print(text)
    lines.append(text)


def step(state, action, st=stats):
    return state + train.predict_delta(model, state, action, *st)


def rmse(e, axis=None):
    return np.sqrt((e ** 2).mean(axis))


val = data[split:]
s, a, s1 = val[:, :S], val[:, S:S + A], val[:, S + A:]

# ---- 0) checkpoint identity ----
xn = (np.c_[s, a] - stats[0]) / stats[1]
yn = (s1 - s - stats[2]) / stats[3]
with torch.no_grad():
    val_loss = float(((model(torch.as_tensor(xn)).numpy() - yn) ** 2).mean())
log = (ROOT / "train_log.txt").read_text().splitlines()
best_line = min((l for l in log if l[:1].isdigit()), key=lambda l: float(l.split(",")[3]))
out("0) Checkpoint")
out(f"   recomputed val one-step loss with dynamics.pt + norm.npz = {val_loss:.6f}; "
    f"train_log best = epoch {best_line.split(',')[0]} ({best_line.split(',')[3]})")
out(f"   recomputed open_loop_rmse@10 = {train.open_loop_rmse(model, data, split, *stats, 10):.6f} "
    f"(train_log summary: {log[-1].split('open-loop@10=')[1].split()[0]})")

# ---- 1) action sensitivity ----
out("\n1) Action sensitivity (validation block)")
dqd_true = s1[:, 6:] - s[:, 6:]
pred = step(s, a)
e_ok = pred[:, 6:] - s1[:, 6:]
out(f"   one-step dqd RMSE (= qd' RMSE), all joints / per joint 1..6:")
out(f"   {'correct actions':28s} {rmse(e_ok):.4f} | " + " ".join(f"{v:.3f}" for v in rmse(e_ok, 0)))
for seed in range(3):
    a_sh = a[np.random.default_rng(seed).permutation(len(a))]
    e_sh = step(s, a_sh)[:, 6:] - s1[:, 6:]
    out(f"   {'shuffled actions, seed ' + str(seed):28s} {rmse(e_sh):.4f} | " + " ".join(f"{v:.3f}" for v in rmse(e_sh, 0)))
out(f"   {'zero dqd (persistence)':28s} {rmse(dqd_true):.4f} | " + " ".join(f"{v:.3f}" for v in rmse(dqd_true, 0)))
# Is the shuffle effect only from wild (a - q)? Same test restricted to rows near the target (|a-q| < 0.05 rad on all joints).
near = np.abs(a - s[:, :6]).max(1) < 0.05
a_sh = a[np.random.default_rng(0).permutation(len(a))]
out(f"   rows with max|a-q| < 0.05 rad ({near.mean():.0%} of val): correct {rmse(e_ok[near]):.4f}, "
    f"shuffled {rmse((step(s, a_sh)[:, 6:] - s1[:, 6:])[near]):.4f}, persistence {rmse(dqd_true[near]):.4f}")

# Multi-step from each goal switch in validation: correct sequence vs old action held vs another window's sequence.
H = 100
act = data[:, S:S + A]
sw = np.where(np.any(act[1:] != act[:-1], axis=1))[0] + 1
sw = sw[(sw >= split) & (sw + H <= len(data))]
true = np.stack([data[i:i + H, S + A:] for i in sw])          # (W, H, 12)
seqs = {
    "correct action sequence": np.stack([act[i:i + H] for i in sw]),
    "old action held (no switch)": np.stack([np.repeat(act[i - 1][None], H, 0) for i in sw]),
    "other window's sequence": np.stack([act[i:i + H] for i in np.roll(sw, 7)]),
}
out(f"\n   multi-step from {len(sw)} goal switches in validation ({H} steps, model fed its own predictions)")
out(f"   {'action sequence':30s} " + " ".join(f"q@{k:<3d}" for k in (10, 25, 50, 100))
    + "   qd@50  dir-ok@50  dir-ok@100")
for name, seq in seqs.items():
    st = data[sw, :S].copy()
    traj = []
    for k in range(H):
        st = step(st, seq[:, k])
        traj.append(st)
    traj = np.stack(traj, 1)
    e = traj - true
    row = " ".join(f"{rmse(e[:, k - 1, :6]):.3f}" for k in (10, 25, 50, 100))
    d = []
    for k in (50, 100):
        dt_, dp = true[:, k - 1, :6] - data[sw, :6], traj[:, k - 1, :6] - data[sw, :6]
        m = np.abs(dt_) > 0.05
        d.append(f"{(np.sign(dt_[m]) == np.sign(dp[m])).mean():.0%} of {m.sum()}")
    out(f"   {name:30s} {row}   {rmse(e[:, 49, 6:]):.3f}  {d[0]:>9s}  {d[1]:>9s}")
out("   (q RMSE in rad over joints and windows; dir-ok = sign of predicted q_k - q_0 matches MuJoCo, joints moving > 0.05 rad)")

# ---- 2) L4-style smoothed actions ----
sm = np.load(OUT / "L3_smooth_rollouts.npz")
ss, sa, sn, goals, kinds = sm["state"], sm["action"], sm["next_state"], sm["goal"], sm["kind"]
E, T = sa.shape[:2]
out(f"\n2) Smoothed-action episodes ({E} x {T} steps, Python MuJoCo 3.17.0 = Unity to "
    f"{float(sm['sanity_max_q_err']):.1e} rad over 200 steps; not added to training data)")
st = ss[:, 0].copy()
pred = []
for k in range(T):
    st = step(st, sa[:, k])
    pred.append(st)
pred = np.stack(pred, 1)
e = pred - sn
pers = ss[:, :1] - sn
out(f"   {'k':>4s} {'q RMSE':>8s} {'q persist':>10s} {'qd RMSE':>8s} {'qd persist':>11s} {'worst ep q RMSE':>16s}")
for k in (10, 50, 100, 200):
    i = k - 1
    out(f"   {k:4d} {rmse(e[:, i, :6]):8.4f} {rmse(pers[:, i, :6]):10.4f} {rmse(e[:, i, 6:]):8.3f} "
        f"{rmse(pers[:, i, 6:]):11.3f} {rmse(e[:, i, :6], 1).max():16.4f}")
out("   per episode q RMSE @50/@100/@200, final |q - goal| MuJoCo vs model:")
for ep in range(E):
    d_true, d_pred = rmse(sn[ep, -1, :6] - goals[ep]), rmse(pred[ep, -1, :6] - goals[ep])
    out(f"   ep{ep} ({kinds[ep]}): " + " / ".join(f"{rmse(e[ep, k - 1, :6]):.3f}" for k in (50, 100, 200))
        + f"   goal dist {d_true:.3f} vs {d_pred:.3f}   max|qd| true {np.abs(sn[ep, -20:, 6:]).max():.2f}"
        f" pred {np.abs(pred[ep, -20:, 6:]).max():.2f} (last 20 steps)")
for k in (50, 100, 200):
    dt_, dp = sn[:, k - 1, :6] - ss[:, 0, :6], pred[:, k - 1, :6] - ss[:, 0, :6]
    m = np.abs(dt_) > 0.05
    wrong = np.argwhere(m & (np.sign(dt_) != np.sign(dp)))
    out(f"   direction @{k}: {(np.sign(dt_[m]) == np.sign(dp[m])).mean():.0%} of {m.sum()} moving joints correct"
        + ("; wrong: " + ", ".join(f"ep{a_}/j{b_ + 1}" for a_, b_ in wrong) if len(wrong) else ""))
gap_train = np.abs(data[:split, S:S + A] - data[:split, :6])
gap_sm = np.abs(sa - ss[:, :, :6]).reshape(-1, 6)
out("   distribution: |a - q| median / 99% train " + f"{np.median(gap_train):.3f} / {np.quantile(gap_train, .99):.3f}"
    + f", smoothed {np.median(gap_sm):.3f} / {np.quantile(gap_sm, .99):.3f} rad; "
    + f"per-step |delta a| > 0: train {np.mean(np.any(np.diff(data[:split, S:S + A], axis=0) != 0, 1)):.2%}, "
    + f"smoothed {np.mean(np.any(np.diff(sa, axis=1) != 0, 2)):.0%} of steps")

ks = np.arange(1, T + 1)
fig, ax = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
for ep in range(E):
    axe = ax.flat[ep]
    for j in range(6):
        line, = axe.plot(ks, sn[ep, :, j], lw=1.5)
        axe.plot(ks, pred[ep, :, j], "--", color=line.get_color(), lw=1)
        axe.plot(ks, sa[ep, :, j], ":", color=line.get_color(), lw=0.8)
    axe.set_title(f"ep{ep}: {kinds[ep]}", fontsize=9)
    axe.grid(alpha=0.3)
fig.suptitle("Smoothed actions: solid = MuJoCo q, dashed = world model q, dotted = action (rad)")
for axe in ax[-1]:
    axe.set_xlabel("step (x 0.002 s)")
fig.tight_layout()
fig.savefig(OUT / "L3_smooth_rollouts.png", dpi=110)

# ---- 3) ONNX / normalization contract ----
out("\n3) ONNX and normalization contract")
om = onnx.load(str(ROOT / "dynamics.onnx"))
ops = sorted({n.op_type for n in om.graph.node})
out(f"   ONNX ops: {ops} -> no normalization inside the graph; input 'x' must be normalized, output 'delta' is normalized")
nj = json.loads((ROOT / "norm.json").read_text())
jstats = tuple(np.array(nj[k], dtype=np.float32) for k in ("x_mean", "x_std", "y_mean", "y_std"))
out("   norm.json vs norm.npz max |diff|: " + ", ".join(
    f"{k} {np.abs(j - n).max():.1e}" for k, j, n in zip(("x_mean", "x_std", "y_mean", "y_std"), jstats, stats))
    + f"; lengths {[len(v) for v in jstats]}")
sess = ort.InferenceSession(str(ROOT / "dynamics.onnx"))


def unity_predict(state, action):
    # What WorldModelDriver.Predict has to do, with norm.json values and the ONNX model, batch 1.
    x = ((np.r_[state, action] - jstats[0]) / jstats[1]).astype(np.float32)[None]
    delta = sess.run(None, {"x": x})[0][0]
    return state + delta * jstats[3] + jstats[2]


worst = 0.0
for ep in range(E):
    su, stt = ss[ep, 0].copy(), ss[ep, 0][None].copy()
    for k in range(T):
        su = unity_predict(su, sa[ep, k])
        stt = step(stt, sa[ep, k][None])
    worst = max(worst, np.abs(su - stt[0]).max())
out(f"   200-step recursion, Unity path (norm.json + ONNX, batch 1) vs Python path (norm.npz + dynamics.pt): "
    f"max |diff| = {worst:.1e}")
x0 = ((np.r_[s[0], a[0]] - jstats[0]) / jstats[1]).astype(np.float32)[None]
d0 = sess.run(None, {"x": x0})[0][0]
out(f"   wrong-contract sizes (val row 0, |qd' error|): correct {np.abs(s[0] + d0 * jstats[3] + jstats[2] - s1[0])[6:].max():.3f}, "
    f"forgot denormalization {np.abs(s[0] + d0 - s1[0])[6:].max():.3f}, "
    f"forgot '+ state' {np.abs(d0 * jstats[3] + jstats[2] - s1[0])[6:].max():.3f} rad/s")

(OUT / "L3_extra_checks.txt").write_text("\n".join(lines) + "\n")
