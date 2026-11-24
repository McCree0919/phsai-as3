"""L5 analysis of the Unity runs (L5Test): loop errors, oscillation, theta = 0 comparison, driver on/off, self-contact
of the IK solutions, and the Unity world-model step vs Python (torch).

python Tools/eval_l5.py   -> TestResults/L5_eval.txt, L5_loop.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "Assets/CA3"))
import train_controller as tc  # noqa: E402

R = PROJECT / "TestResults"
lines = []


def out(text=""):
    print(text)
    lines.append(text)


def load(tag):
    d = np.loadtxt(R / f"L5_loop{tag}.csv", delimiter=",", skiprows=1, dtype=np.float64)
    return dict(phase=d[:, 0].astype(int), t=d[:, 2:5], goal=d[:, 5:11], ikerr=d[:, 11], q=d[:, 12:18], qd=d[:, 18:24],
                ctrl=d[:, 24:30], ee=d[:, 30:33], dist=d[:, 33])


names = ["near home", "left", "right", "high", "low", "far", "unreachable"]
runs = {"L4.2": load(""), "theta=0": load("_theta0"), "L4.2, driver off": load("_nodriver")}

out("L5 full loop in Unity (TrackingTarget -> IK -> WorldModelController -> MuJoCo), 2 s per target")
out("per target: EE distance at the end (mm) / time until the EE is within 5 mm (s) / max overshoot past the final "
    "EE position along the move (mm) / last 0.5 s: EE peak-to-peak (mm), max joint p2p (rad), max |qd| (rad/s)")
for name, r in runs.items():
    out(f"  {name}")
    for k, nm in enumerate(names):
        i = np.where(r["phase"] == k)[0]
        dist, ee = r["dist"][i], r["ee"][i]
        within = np.where(dist < 0.005)[0]
        t5 = f"{within[0] * 0.002:.2f}" if len(within) else "  -  "
        move = ee[-1] - ee[0]
        u = move / (np.linalg.norm(move) + 1e-9)
        over = max(0.0, ((ee - ee[-1]) @ u).max())
        last = i[-250:]
        p2p_ee = np.linalg.norm(r["ee"][last].max(0) - r["ee"][last].min(0))
        p2p_q = (r["q"][last].max(0) - r["q"][last].min(0)).max()
        out(f"    #{k} {nm:11s} dist {1000 * dist[-1]:7.2f}  t<5mm {t5}  overshoot {1000 * over:5.2f}  "
            f"p2p EE {1000 * p2p_ee:.3f} q {p2p_q:.4f}  |qd| {np.abs(r['qd'][last]).max():.3f}  "
            f"IK err {1000 * r['ikerr'][i][-1]:.2f}  joint err RMSE {np.sqrt(((r['q'][i][-1] - r['goal'][i][-1]) ** 2).mean()):.4f}")

a, b = runs["L4.2"], runs["L4.2, driver off"]
out(f"\nWorldModelDriver on vs off (same scene, L4.2): max |q difference| over all {len(a['q'])} steps = "
    f"{np.abs(a['q'] - b['q']).max():.2e} rad, max |ctrl difference| = {np.abs(a['ctrl'] - b['ctrl']).max():.2e}")

# self-contact of the IK solutions and of the real final poses (Python MuJoCo, same lite6.xml)
m = mujoco.MjModel.from_xml_path(str(PROJECT / "Robots/ufactory_lite6/lite6.xml"))
d = mujoco.MjData(m)


def contacts(q):
    d.qpos[:6] = q
    mujoco.mj_forward(m, d)
    return sorted({(mujoco.mj_id2name(m, 1, m.geom_bodyid[c.geom1]), mujoco.mj_id2name(m, 1, m.geom_bodyid[c.geom2]))
                   for c in d.contact[:d.ncon]})


out("\nSelf-contact check (MuJoCo contacts at the pose):")
for l in (R / "L5_ik.txt").read_text().splitlines():
    if l.strip().startswith("ikq,"):
        _, nm, *q = l.strip().split(",")
        out(f"  IK test solution {nm:11s}: {contacts(np.array(q, float)) or 'none'}")
for name in ("L4.2", "theta=0"):
    r = runs[name]
    for k, nm in enumerate(names):
        i = np.where(r["phase"] == k)[0]
        c = [contacts(q) for q in r["q"][i][::50]]
        out(f"  {name:8s} loop #{k} {nm:11s}: steps with contacts (every 50th checked) {sum(bool(x) for x in c)}/{len(c)} "
            f"{sorted(set(p for x in c for p in x)) or ''}")

# world model: Unity Predict (Inference Engine, ONNX) vs Python torch, same inputs
w = np.loadtxt(R / "L5_wm_unity.csv", delimiter=",", dtype=np.float32)
s, act, pu, ms = w[:, :12], w[:, 12:18], w[:, 18:30], w[:, 30]
model, stats = tc.build_dynamics(), tc.load_norm()
pp = np.array([tc.step_dynamics(model, s[k], act[k], *stats) for k in range(len(w))])
# real next state: the logged state of the following step
qa = a["q"], a["qd"]
out(f"\nWorld model in Unity (dynamics_multistep.onnx + norm_multistep.json, Inference Engine CPU), {len(w)} samples:")
out(f"  max |Unity - Python| next state: q {np.abs(pu - pp)[:, :6].max():.2e} rad, qd {np.abs(pu - pp)[:, 6:].max():.2e} rad/s; "
    f"max |Unity - Python| delta (next - state) relative {np.abs((pu - s) - (pp - s)).max() / np.abs(pp - s).max():.2e}")
out(f"  Predict time in Unity: median {np.median(ms):.3f} ms, max {ms.max():.3f} ms")

# plot EE distance over time
fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for name, st in (("L4.2", "-"), ("theta=0", "--")):
    r = runs[name]
    t = np.arange(len(r["dist"])) * 0.002
    ax[0].plot(t, 1000 * r["dist"], st, label=name)
    ax[1].plot(t, r["q"], st, lw=1)
ax[0].set_yscale("log")
ax[0].set_ylabel("EE distance to target (mm)")
ax[0].legend()
ax[1].set_ylabel("q (rad), solid = L4.2, dashed = theta=0")
ax[1].set_xlabel("time (s); the target moves every 2 s: " + ", ".join(names))
for x in ax:
    x.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(R / "L5_loop.png", dpi=110)
(R / "L5_eval.txt").write_text("\n".join(lines) + "\n")
