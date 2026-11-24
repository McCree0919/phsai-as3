"""L3 evaluation of the trained world model on the validation block (last 20% of rollout.csv).

python Tools/eval_world_model.py   -> TestResults/L3_eval.txt, L3_loss.png, L3_multistep.png, L3_rollouts.png
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "TestResults"
spec = importlib.util.spec_from_file_location("train", PROJECT / "Assets/CA3/train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
S, A, ROOT = train.STATE, train.ACTION, train.ROOT
DT, H = 0.002, 200  # multi-step horizon: 200 steps = 0.4 s

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


def step(state, action):
    return state + train.predict_delta(model, state, action, *stats)


val = data[split:]
s, a, s1 = val[:, :S], val[:, S:S + A], val[:, S + A:]
out(f"validation rows [{split}, {len(data)}) = {len(val)} steps = {len(val) * DT:.1f} s (contiguous, unseen in training)")

# ---- A/B: one-step prediction vs baselines ----
preds = {
    "world model": step(s, a),
    "persistence (s' = s)": s.copy(),
    "constant velocity (q' = q + dt*qd, qd' = qd)": np.c_[s[:, :6] + DT * s[:, 6:], s[:, 6:]],
}
s_std = stats[1][:S]
out("\nA/B. One-step prediction (RMSE / MAE over all joints; q in rad, qd in rad/s; overall = RMSE of state/std)")
out(f"{'predictor':46s} {'q RMSE':>10s} {'q MAE':>10s} {'qd RMSE':>9s} {'qd MAE':>9s} {'overall':>9s}")
for name, p in preds.items():
    e = p - s1
    out(f"{name:46s} {np.sqrt((e[:, :6] ** 2).mean()):10.2e} {np.abs(e[:, :6]).mean():10.2e} "
        f"{np.sqrt((e[:, 6:] ** 2).mean()):9.4f} {np.abs(e[:, 6:]).mean():9.4f} {np.sqrt(((e / s_std) ** 2).mean()):9.2e}")
e_m, e_p = preds["world model"] - s1, preds["persistence (s' = s)"] - s1
out("per joint qd RMSE  model: " + " ".join(f"{v:.3f}" for v in np.sqrt((e_m[:, 6:] ** 2).mean(0)))
    + " | persistence: " + " ".join(f"{v:.3f}" for v in np.sqrt((e_p[:, 6:] ** 2).mean(0))))
r2 = 1 - (e_m[:, 6:] ** 2).sum(0) / ((s1[:, 6:] - s[:, 6:] - (s1[:, 6:] - s[:, 6:]).mean(0)) ** 2).sum(0)
out("R^2 of predicted dqd per joint: " + " ".join(f"{v:.2f}" for v in r2))
out(f"teacher metric joint_rmse (one-step, q vector norm): {train.joint_rmse(preds['world model'], s1):.6f} rad")

# ---- C: multi-step open-loop prediction with the recorded actions ----
starts = np.arange(split, len(data) - H, 25)
sw = np.where(np.any(data[1:, S:S + A] != data[:-1, S:S + A], axis=1))[0] + 1
has_switch = np.array([np.any((sw > i) & (sw < i + H)) for i in starts])
st = data[starts, :S].copy()
err_q, err_qd, per_q, per_qd = [], [], [], []
for k in range(H):
    st = step(st, data[starts + k, S:S + A])
    true = data[starts + k, S + A:]
    e = st - true
    pe = data[starts, :S] - true
    err_q.append(np.sqrt((e[:, :6] ** 2).mean(1)))
    err_qd.append(np.sqrt((e[:, 6:] ** 2).mean(1)))
    per_q.append(np.sqrt((pe[:, :6] ** 2).mean(1)))
    per_qd.append(np.sqrt((pe[:, 6:] ** 2).mean(1)))
err_q, err_qd, per_q, per_qd = map(np.array, (err_q, err_qd, per_q, per_qd))  # (H, windows)
out(f"\nC. Multi-step open-loop prediction: {len(starts)} windows of {H} steps, recorded actions, own predictions fed back")
out(f"{'k (steps / s)':>16s} {'q RMSE model':>13s} {'q persist':>10s} {'qd RMSE model':>14s} {'qd persist':>11s}"
    f" {'q RMSE no-switch':>17s} {'q RMSE switch':>14s} {'worst q RMSE':>13s}")
for k in (1, 10, 25, 50, 100, 200):
    i = k - 1
    out(f"{k:5d} / {k * DT:5.3f}s {err_q[i].mean():13.4f} {per_q[i].mean():10.4f} {err_qd[i].mean():14.3f} {per_qd[i].mean():11.3f}"
        f" {err_q[i][~has_switch].mean():17.4f} {err_q[i][has_switch].mean():14.4f} {err_q[i].max():13.4f}")
for h in (10, 50, 100):
    out(f"teacher open_loop_rmse@{h}: {train.open_loop_rmse(model, data, split, *stats, h):.6f} rad")
out(f"windows containing a goal switch: {has_switch.sum()}/{len(starts)}")

ks = np.arange(1, H + 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for j, (m, p, lab) in enumerate([(err_q, per_q, "q RMSE (rad)"), (err_qd, per_qd, "qd RMSE (rad/s)")]):
    ax[j].plot(ks, m.mean(1), label="world model (all)")
    ax[j].plot(ks, m[:, ~has_switch].mean(1), "--", label="world model, no goal switch")
    ax[j].plot(ks, m[:, has_switch].mean(1), ":", label="world model, goal switch in window")
    ax[j].plot(ks, p.mean(1), color="gray", label="persistence")
    ax[j].set_xlabel("prediction step (x 0.002 s)")
    ax[j].set_ylabel(lab)
    ax[j].set_yscale("log")
    ax[j].grid(alpha=0.3)
ax[0].legend(fontsize=8)
fig.suptitle("L3 open-loop multi-step error on validation block")
fig.tight_layout()
fig.savefig(OUT / "L3_multistep.png", dpi=120)

# Example windows: right after a goal switch, a settled one, and the worst one.
val_sw = sw[(sw >= split) & (sw < len(data) - H)]
examples = {"start at goal switch": int(val_sw[0]),
            "no goal switch in window": int(starts[~has_switch][0]),
            "worst window": int(starts[np.argmax(err_q[-1])])}
fig, ax = plt.subplots(len(examples), 1, figsize=(10, 8), sharex=True)
for r, (title, i0) in enumerate(examples.items()):
    st = data[i0, :S].copy()[None]
    traj = []
    for k in range(H):
        st = step(st, data[i0 + k, S:S + A][None])
        traj.append(st[0])
    traj = np.array(traj)
    true = data[i0:i0 + H, S + A:]
    for j in range(6):
        line, = ax[r].plot(ks, true[:, j], lw=1.5)
        ax[r].plot(ks, traj[:, j], "--", color=line.get_color(), lw=1)
    ax[r].set_title(f"{title} (row {i0}): solid = MuJoCo, dashed = world model; q1..q6 (rad)", fontsize=9)
    ax[r].grid(alpha=0.3)
ax[-1].set_xlabel("prediction step (x 0.002 s)")
fig.tight_layout()
fig.savefig(OUT / "L3_rollouts.png", dpi=120)
out("examples: " + ", ".join(f"{k}=row {v}" for k, v in examples.items()))

# ---- training curves ----
log = [l.split(",") for l in (ROOT / "train_log.txt").read_text().splitlines()[2:] if l[:1].isdigit()]
log = np.array(log, dtype=float)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(log[:, 0], log[:, 1], label="train one-step (normalized MSE)")
ax.plot(log[:, 0], log[:, 3], label="val one-step (normalized MSE)")
ax.plot(log[:, 0], log[:, 2], label="train multi-step@20")
ax.set_xlabel("epoch")
ax.set_yscale("log")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "L3_loss.png", dpi=120)

# ---- ONNX export check ----
sess = ort.InferenceSession(str(ROOT / "dynamics.onnx"))
i, o = sess.get_inputs()[0], sess.get_outputs()[0]
xn = ((np.c_[s, a] - stats[0]) / stats[1]).astype(np.float32)[:1000]
diff = max(np.abs(sess.run(None, {i.name: xn[k:k + 1]})[0] - model(torch.as_tensor(xn[k:k + 1])).detach().numpy()).max()
           for k in range(0, 1000, 10))
out(f"\nONNX: input {i.name} {i.shape}, output {o.name} {o.shape}, opset "
    f"{__import__('onnx').load(str(ROOT / 'dynamics.onnx')).opset_import[0].version}, max |onnx - torch| = {diff:.2e}")

(OUT / "L3_eval.txt").write_text("\n".join(lines) + "\n")
