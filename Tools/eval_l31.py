"""L3.1 evaluation: old L3 model (dynamics.pt) vs new model trained with smoothed-action data (dynamics_smooth.pt).

python Tools/eval_l31.py             -> TestResults/L31_eval.txt, L31_multistep.png, L31_smooth_rollouts.png
python Tools/eval_l31.py multistep   -> also dynamics_multistep.pt (multi-step checkpoint rule), files L31ms_*
Test sets (none used for training):
  step val      rollout.csv rows [56000, 70000)
  smooth val    rollout_smooth.csv rows [40000, 50000)
  test seed123  TestResults/L3_smooth_rollouts.npz       (8 episodes, the L3 extra-check set)
  test seed2027 TestResults/L31_smooth_test_seed2027.npz (16 episodes, independent seed)
"""
import importlib.util
import json
import sys
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
H = 200
lines = []


def out(text=""):
    print(text)
    lines.append(text)


def load(pt, nz):
    m = train.build_model()
    m.load_state_dict(torch.load(ROOT / pt, weights_only=True))
    m.eval()
    z = np.load(ROOT / nz)
    return m, (z["x_mean"], z["x_std"], z["y_mean"], z["y_std"])


MODELS = {"old L3": load("dynamics.pt", "norm.npz"), "new L3.1": load("dynamics_smooth.pt", "norm_smooth.npz")}
STYLES = {"old L3": ":", "new L3.1": "--", "L3.1 ms": "-"}
EXPORT, PREFIX = "smooth", "L31_"
if "multistep" in sys.argv:
    MODELS["L3.1 ms"] = load("dynamics_multistep.pt", "norm_multistep.npz")
    EXPORT, PREFIX = "multistep", "L31ms_"
NEWEST = list(MODELS)[-1]


def step(M, s, a):
    return s + train.predict_delta(M[0], s, a, *M[1])


def rollout(M, s0, actions):  # s0 (N, 12), actions (N, T, 6) -> (N, T, 12)
    st, traj = s0.copy(), []
    for k in range(actions.shape[1]):
        st = step(M, st, actions[:, k])
        traj.append(st)
    return np.stack(traj, 1)


def rmse(e, axis=None):
    return np.sqrt((e ** 2).mean(axis))


def windows(data, start, every=25):
    i0 = np.arange(start, len(data) - H, every)
    idx = i0[:, None] + np.arange(H)
    return data[i0, :S], data[idx, S:S + A], data[idx, S + A:]


def scores(pred, true, s0, ks=(50, 100, 200)):
    r = {}
    for k in ks:
        e = pred[:, k - 1] - true[:, k - 1]
        dt_, dp = true[:, k - 1, :6] - s0[:, :6], pred[:, k - 1, :6] - s0[:, :6]
        m = np.abs(dt_) > 0.05
        r[k] = dict(q=rmse(e[:, :6]), qd=rmse(e[:, 6:]), dir=(np.sign(dt_[m]) == np.sign(dp[m])).mean(),
                    ratio=(dp[m] * dt_[m]).sum() / (dt_[m] ** 2).sum())  # displacement ratio < 1 = lagging
    return r


step_data = train.load_rollout("rollout.csv")
smooth_data = train.load_rollout("rollout_smooth.csv")
sp_step, sp_smooth = int(len(step_data) * 0.8), int(len(smooth_data) * 0.8)
tests = {name: np.load(OUT / f) for name, f in
         [("test seed123", "L3_smooth_rollouts.npz"), ("test seed2027", "L31_smooth_test_seed2027.npz")]}

# ---- 0) data coverage ----
out("0) Action coverage: share of steps where the action changes, |a - q| quantiles 50/90/99% (rad)")
cover = {"rollout.csv train": step_data[:sp_step], "rollout_smooth.csv train": smooth_data[:sp_smooth]}
for name, d in cover.items():
    a, q = d[:, S:S + A], d[:, :6]
    out(f"   {name:28s} changes {np.mean(np.any(a[1:] != a[:-1], 1)):6.1%}   |a-q| "
        + " / ".join(f"{v:.3f}" for v in np.quantile(np.abs(a - q), [.5, .9, .99])))
for name, t in tests.items():
    a, q = t["action"], t["state"][..., :6]
    out(f"   {name:28s} changes {np.mean(np.any(np.diff(a, axis=1) != 0, 2)):6.1%}   |a-q| "
        + " / ".join(f"{v:.3f}" for v in np.quantile(np.abs(a - q), [.5, .9, .99])))

# ---- A) original step data ----
out("\nA) Step-target validation (rollout.csv rows [56000, 70000))")
v = step_data[sp_step:]
s, a, s1 = v[:, :S], v[:, S:S + A], v[:, S + A:]
out(f"   {'one-step':10s} {'q RMSE':>9s} {'qd RMSE':>8s}")
out(f"   {'persist':10s} {rmse((s - s1)[:, :6]):9.2e} {rmse((s - s1)[:, 6:]):8.4f}")
for name, M in MODELS.items():
    e = step(M, s, a) - s1
    out(f"   {name:10s} {rmse(e[:, :6]):9.2e} {rmse(e[:, 6:]):8.4f}")
s0, acts, true = windows(step_data, sp_step)
out(f"   multi-step ({len(s0)} windows): q RMSE @10/50/100/200 | qd RMSE @10/50/100/200 | teacher open_loop_rmse @10/50/100")
for name, M in MODELS.items():
    p = rollout(M, s0, acts)
    out(f"   {name:10s} " + " ".join(f"{rmse((p - true)[:, k - 1, :6]):.4f}" for k in (10, 50, 100, 200)) + " | "
        + " ".join(f"{rmse((p - true)[:, k - 1, 6:]):.3f}" for k in (10, 50, 100, 200)) + " | "
        + " ".join(f"{train.open_loop_rmse(M[0], step_data, sp_step, *M[1], h):.4f}" for h in (10, 50, 100)))

# ---- B) smoothed actions ----
out("\nB) Smoothed-action sets (never trained on). q / qd RMSE, direction ok, displacement ratio (pred/true, <1 = lag)")
sets = {"smooth val (Unity) windows": windows(smooth_data, sp_smooth)}
for name, t in tests.items():
    n = len(t["action"])
    for part, sl in [("A raw=goal", slice(0, n // 2)), ("B random policy", slice(n // 2, n))]:
        sets[f"{name} {part}"] = (t["state"][sl, 0], t["action"][sl], t["next_state"][sl])
curves = {}
for name, (s0, acts, true) in sets.items():
    out(f"   {name} ({len(s0)} x {H} steps)")
    for mname, M in MODELS.items():
        p = rollout(M, s0, acts)
        curves[(name, mname)] = (rmse((p - true)[..., :6], (0, 2)), rmse((p - true)[..., 6:], (0, 2)))
        r = scores(p, true, s0)
        out(f"     {mname:9s} " + "   ".join(
            f"@{k}: q {r[k]['q']:.3f} qd {r[k]['qd']:.2f} dir {r[k]['dir']:.0%} ratio {r[k]['ratio']:.2f}" for k in r))
out("   final distance to goal at step 200 (rad, RMSE over joints): mean |pred - true|, mean (pred - true)")
for name, t in tests.items():
    n = len(t["action"])
    d_true = rmse(t["next_state"][:, -1, :6] - t["goal"], 1)
    for mname, M in MODELS.items():
        d_pred = rmse(rollout(M, t["state"][:, 0], t["action"])[:, -1, :6] - t["goal"], 1)
        out(f"   {name} {mname:9s} A: {np.abs(d_pred - d_true)[:n // 2].mean():.3f} / {(d_pred - d_true)[:n // 2].mean():+.3f}"
            f"   B: {np.abs(d_pred - d_true)[n // 2:].mean():.3f} / {(d_pred - d_true)[n // 2:].mean():+.3f}"
            f"   (true dist A {d_true[:n // 2].mean():.3f}, B {d_true[n // 2:].mean():.3f})")
out("   one-step on smooth val: " + ", ".join(
    f"{mname} q {rmse((step(M, smooth_data[sp_smooth:, :S], smooth_data[sp_smooth:, S:S + A]) - smooth_data[sp_smooth:, S + A:])[:, :6]):.2e}"
    f" qd {rmse((step(M, smooth_data[sp_smooth:, :S], smooth_data[sp_smooth:, S:S + A]) - smooth_data[sp_smooth:, S + A:])[:, 6:]):.4f}"
    for mname, M in MODELS.items()) + f", persistence qd {rmse(np.diff(smooth_data[sp_smooth:, [*range(6, 12)]], axis=0)):.4f}")

# ---- C) action sensitivity ----
out("\nC) Action sensitivity: one-step qd RMSE with correct vs shuffled actions; multi-step from goal switches")
for vname, d, sp in [("step val", step_data, sp_step), ("smooth val", smooth_data, sp_smooth)]:
    s, a, s1 = d[sp:, :S], d[sp:, S:S + A], d[sp:, S + A:]
    a_sh = a[np.random.default_rng(0).permutation(len(a))]
    out(f"   {vname:10s} " + "   ".join(
        f"{mname}: correct {rmse((step(M, s, a) - s1)[:, 6:]):.4f} shuffled {rmse((step(M, s, a_sh) - s1)[:, 6:]):.4f}"
        for mname, M in MODELS.items()))
act = step_data[:, S:S + A]
sw = np.where(np.any(act[1:] != act[:-1], axis=1))[0] + 1
sw = sw[(sw >= sp_step) & (sw + 100 <= len(step_data))]
true = np.stack([step_data[i:i + 100, S + A:] for i in sw])
seqs = {"correct": np.stack([act[i:i + 100] for i in sw]),
        "old action held": np.stack([np.repeat(act[i - 1][None], 100, 0) for i in sw]),
        "other window": np.stack([act[i:i + 100] for i in np.roll(sw, 7)])}
for mname, M in MODELS.items():
    parts = []
    for qname, seq in seqs.items():
        r = scores(rollout(M, step_data[sw, :S], seq), true, step_data[sw, :S], ks=(50,))[50]
        parts.append(f"{qname} q@50 {r['q']:.3f} dir {r['dir']:.0%}")
    out(f"   {len(sw)} switches, {mname:9s} " + " | ".join(parts))

# ---- D) export ----
out(f"\nD) {NEWEST} export (dynamics_{EXPORT}.onnx + norm_{EXPORT}.json)")
om = onnx.load(str(ROOT / f"dynamics_{EXPORT}.onnx"))
sess = ort.InferenceSession(str(ROOT / f"dynamics_{EXPORT}.onnx"))
i, o = sess.get_inputs()[0], sess.get_outputs()[0]
nj = json.loads((ROOT / f"norm_{EXPORT}.json").read_text())
js = tuple(np.array(nj[k], dtype=np.float32) for k in ("x_mean", "x_std", "y_mean", "y_std"))
M = MODELS[NEWEST]
out(f"   input {i.name} {i.shape}, output {o.name} {o.shape}, opset {om.opset_import[0].version}, "
    f"ops {sorted({n.op_type for n in om.graph.node})} (no normalization in the graph)")
out(f"   norm_{EXPORT}.json vs .npz max |diff| " + ", ".join(
    f"{k} {np.abs(a_ - b_).max():.1e}" for k, a_, b_ in zip(("x_mean", "x_std", "y_mean", "y_std"), js, M[1])))
xn = ((smooth_data[sp_smooth:sp_smooth + 1000, :S + A] - js[0]) / js[1]).astype(np.float32)
diff = max(np.abs(sess.run(None, {"x": xn[k:k + 1]})[0] - M[0](torch.as_tensor(xn[k:k + 1])).detach().numpy()).max()
           for k in range(0, 1000, 10))
t = tests["test seed2027"]
worst = 0.0
for ep in range(len(t["action"])):
    su = t["state"][ep, 0].copy()
    for k in range(H):  # Unity path: x = ([s, a] - x_mean) / x_std; next = s + delta * y_std + y_mean
        x = ((np.r_[su, t["action"][ep, k]] - js[0]) / js[1]).astype(np.float32)[None]
        su = su + sess.run(None, {"x": x})[0][0] * js[3] + js[2]
    worst = max(worst, np.abs(su - rollout(M, t["state"][ep:ep + 1, 0], t["action"][ep:ep + 1])[0, -1]).max())
out(f"   one-step max |onnx - torch| (normalized delta) = {diff:.1e}; 200-step Unity-path vs Python recursion max |diff| = {worst:.1e}")

# ---- plots ----
ks = np.arange(1, H + 1)
names = list(sets)
fig, ax = plt.subplots(2, len(names), figsize=(4 * len(names), 6.5), sharex=True)
for c, name in enumerate(names):
    for mname in MODELS:
        style = STYLES[mname]
        qc, qdc = curves[(name, mname)]
        ax[0, c].plot(ks, qc, style, label=mname)
        ax[1, c].plot(ks, qdc, style, label=mname)
    ax[0, c].set_title(name, fontsize=9)
    ax[1, c].set_xlabel("prediction step (x 0.002 s)")
    for r in range(2):
        ax[r, c].grid(alpha=0.3)
ax[0, 0].set_ylabel("q RMSE (rad)")
ax[1, 0].set_ylabel("qd RMSE (rad/s)")
ax[0, 0].legend(fontsize=8)
fig.suptitle("L3.1 open-loop error: " + " vs ".join(MODELS))
fig.tight_layout()
fig.savefig(OUT / f"{PREFIX}multistep.png", dpi=110)

t = tests["test seed2027"]
eps = [0, 1, 2, 3, 8, 9, 10, 11]
preds = {m: rollout(M, t["state"][eps, 0], t["action"][eps]) for m, M in MODELS.items()}
fig, ax = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
for r, ep in enumerate(eps):
    axe = ax.flat[r]
    for j in range(6):
        line, = axe.plot(ks, t["next_state"][ep, :, j], lw=1.6)
        for mname in (NEWEST, "old L3"):
            axe.plot(ks, preds[mname][r, :, j], "--" if mname == NEWEST else ":", color=line.get_color(), lw=1)
    axe.set_title(f"seed2027 ep{ep}: {t['kind'][ep]}", fontsize=9)
    axe.grid(alpha=0.3)
for axe in ax[-1]:
    axe.set_xlabel("step (x 0.002 s)")
fig.suptitle(f"Independent smoothed-action test: solid = MuJoCo q, dashed = {NEWEST} model, dotted = old L3 model (rad)")
fig.tight_layout()
fig.savefig(OUT / f"{PREFIX}smooth_rollouts.png", dpi=110)

(OUT / f"{PREFIX}eval.txt").write_text("\n".join(lines) + "\n")
