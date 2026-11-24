"""L4.1 real-MuJoCo comparison: L4 controller vs L4.1 controller vs theta = 0 vs direct goal.

python Tools/eval_l41.py   -> TestResults/L41_eval.txt, L41_cma.png, L41_tracking.png
Same 20 goals as L4 (seed 2026), rest start (1 s settle at the start pose), 2 s test, Unity timing for all
controllers: the action from the state before step t is used by step t + 2 (ctrl holds the start pose until then).
"""
import json
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

OUT = PROJECT / "TestResults"
model = tc.build_dynamics()
stats = tc.load_norm()
mjm = mujoco.MjModel.from_xml_path(str(PROJECT / "Robots/ufactory_lite6/lite6.xml"))
mjm.opt.timestep = 0.002
data = mujoco.MjData(mjm)
T, DELAY = 1000, 2
lines = []


def out(text=""):
    print(text)
    lines.append(text)


def load(name):
    c = json.loads((tc.ROOT / name).read_text())
    return np.array(c["W"], dtype=np.float32).reshape(tc.ACTION, tc.FEAT), np.array(c["b"], dtype=np.float32)


ZERO = (np.zeros((tc.ACTION, tc.FEAT), np.float32), np.zeros(tc.ACTION, np.float32))
CONTROLLERS = {"L4 learned": load("controller.json"), "L4.1 learned": load("controller_l41.json"),
               "theta=0": ZERO, "direct goal": None}


def rmse(e, axis=None):
    return np.sqrt((e ** 2).mean(axis))


def state():
    return np.r_[data.qpos[:6], data.qvel[:6]].astype(np.float32)


def settle(start):
    mujoco.mj_resetData(mjm, data)
    data.qpos[:6] = start
    data.ctrl[:] = start
    for _ in range(500):
        mujoco.mj_step(mjm, data)
    return state()


def run_real(params, start, goal):
    s0 = settle(start)
    prev, queue = s0[:6].copy(), [start.copy() for _ in range(DELAY)]
    traj, acts = [], []
    for _ in range(T):
        if params is None:  # direct goal: ArmJointController path, same 2-step timing
            prev = goal.copy()
            queue.append(prev)
            applied = queue.pop(0)
        else:
            prev, applied = tc.controller_step(state(), goal, *params, prev, queue)
        acts.append(prev)
        data.ctrl[:] = applied
        mujoco.mj_step(mjm, data)
        traj.append(state())
    return s0, np.array(traj), np.array(acts)


def run_wm(params, s0, goal, start):
    st, prev, queue = s0.copy(), s0[:6].copy(), [start.copy() for _ in range(DELAY)]
    traj = []
    for _ in range(tc.HORIZON):
        prev, applied = tc.controller_step(st, goal, *params, prev, queue)
        st = tc.step_dynamics(model, st, applied, *stats)
        traj.append(st)
    return np.array(traj)


def metrics(s0, traj, acts, goal):
    q, qd = traj[:, :6], traj[:, 6:]
    d = goal - s0[:6]
    mv = np.abs(d) > 0.05
    over = np.where(mv, np.clip(((q - goal) * np.sign(d)).max(0), 0, None), 0).max()
    a = np.vstack([s0[:6], acts[:tc.HORIZON]])
    terms = [((q[:tc.HORIZON] - goal) ** 2).sum(1).mean(), (qd[:tc.HORIZON] ** 2).sum(1).mean(),
             tc.ACTION_SMOOTH_PENALTY * (np.diff(a, axis=0) ** 2).sum(1).mean()]
    return dict(e200=rmse(q[199] - goal), e1000=rmse(q[-1] - goal), emax=np.abs(q[-1] - goal).max(),
                qd_end=np.abs(qd[-100:]).max(), over=over, p2p=(q[-250:].max(0) - q[-250:].min(0)).max(), terms=terms)


rng = np.random.default_rng(2026)
cases = [(tc.sample_goal(rng), tc.sample_goal(rng)) for _ in range(20)]
res = {name: [] for name in CONTROLLERS}
rows, wm_err = [], []
for k, (start, goal) in enumerate(cases):
    trs = {}
    for name, params in CONTROLLERS.items():
        s0, tr, acts = run_real(params, start, goal)
        res[name].append(metrics(s0, tr, acts, goal))
        trs[name] = tr
    wm = run_wm(CONTROLLERS["L4.1 learned"], s0, goal, start)
    wm_err.append((rmse(wm[-1, :6] - trs["L4.1 learned"][199, :6]), rmse(wm[-1, :6] - goal), rmse(trs["L4.1 learned"][199, :6] - goal)))
    rows.append((k, s0, goal, trs, wm))
reach = [k for k in range(20) if res["direct goal"][k]["e1000"] <= 0.05]
unreach = [k for k in range(20) if k not in reach]

out("L4.1 real MuJoCo test: 20 goals (seed 2026), rest start, 2 s, Unity 2-step timing for every controller")
out(f"goals the direct baseline cannot reach within 0.05 rad (self-collision, joint5 goal > ~1.77): {unreach}")
for label, idx in (("all 20 goals", list(range(20))), (f"{len(reach)} reachable goals", reach)):
    out(f"\n{label}: mean over goals (errors = RMSE over joints of q - goal, rad)")
    out(f"   {'controller':14s} {'err@0.4s':>9s} {'err@2s':>8s} {'median@2s':>9s} {'max jnt@2s':>10s} {'|qd|end':>8s} "
        f"{'overshoot':>9s} {'osc p2p':>8s} | {'|q-g|^2':>8s} {'|qd|^2':>7s} {'0.1|da|^2':>9s} {'fit(0.005)':>10s} {'fit(0.05)':>9s}")
    for name in CONTROLLERS:
        r = [res[name][k] for k in idx]
        g = lambda key: np.array([x[key] for x in r])
        tm = np.array([x["terms"] for x in r]).mean(0)
        out(f"   {name:14s} {g('e200').mean():9.3f} {g('e1000').mean():8.3f} {np.median(g('e1000')):9.3f} {g('emax').mean():10.3f} "
            f"{g('qd_end').mean():8.2f} {g('over').mean():9.3f} {g('p2p').mean():8.3f} | {tm[0]:8.3f} {tm[1]:7.2f} {tm[2]:9.5f} "
            f"{tm[0] + 0.005 * tm[1] + tm[2]:10.3f} {tm[0] + 0.05 * tm[1] + tm[2]:9.3f}")
out("   (reward terms are per-step means over the first 200 steps; fit = |q-g|^2 + VEL_PENALTY |qd|^2 + 0.1|da|^2)")

out("\nPer-goal final error at 2 s (RMSE over joints, rad) / |qd| in the last 0.2 s (rad/s)")
out("   goal " + " ".join(f"{n:>18s}" for n in CONTROLLERS))
for k in range(20):
    out(f"   #{k:2d}{'*' if k in unreach else ' '} " + " ".join(
        f"{res[n][k]['e1000']:9.3f} / {res[n][k]['qd_end']:5.2f}  " for n in CONTROLLERS))
out("   (* = not reachable by the direct baseline)")

w = np.array(wm_err)
out(f"\nWorld model (with the 2-step delay) vs real MuJoCo, L4.1 controller, step 200: end q RMSE mean {w[:, 0].mean():.3f} "
    f"max {w[:, 0].max():.3f}; predicted dist-to-goal {w[:, 1].mean():.3f} vs real {w[:, 2].mean():.3f}")
b = CONTROLLERS["L4.1 learned"][1]
Wq = CONTROLLERS["L4.1 learned"][0][:, :6]
ss = np.linalg.solve(np.eye(6) - Wq, b)  # static offset if q followed raw exactly at rest: e = W_q e + b
out(f"b (L4.1) = [{' '.join(f'{v:+.3f}' for v in b)}], |b| = {np.linalg.norm(b):.3f} (L4: {np.linalg.norm(CONTROLLERS['L4 learned'][1]):.3f});"
    f" implied static offset (I - W_q)^-1 b = [{' '.join(f'{v:+.3f}' for v in ss)}] rad")
fin = np.array([rows[k][3]["L4.1 learned"][-1, :6] - rows[k][2] for k in reach])
out(f"measured mean (q - goal) at 2 s over reachable goals, L4.1: [{' '.join(f'{v:+.3f}' for v in fin.mean(0))}] rad")

# ---- plots ----
cl = (tc.ROOT / "controller_l41_log.txt").read_text().splitlines()
cl = np.array([l.split(",") for l in cl[2:-1]], dtype=float)
zero = float(cl_text.split("theta=0 held-out fitness=")[1].split()[0]) if (cl_text := (tc.ROOT / "controller_l41_log.txt").read_text()) else 0
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(cl[:, 0], cl[:, 1], label="best fitness (iteration's episodes)")
ax[0].plot(cl[:, 0], cl[:, 2], label="median fitness")
ax[0].plot(cl[:, 0], cl[:, 3], label="mean theta, held-out episodes")
ax[0].axhline(zero, color="gray", ls="--", label="theta = 0, held-out")
ax[0].set_xlabel("CMA-ES iteration")
ax[0].set_ylabel("fitness (lower is better)")
ax[0].legend(fontsize=8)
for j, lab in ((4, "position"), (5, "velocity (x0.005)"), (6, "action change")):
    ax[1].plot(cl[:, 0], cl[:, j], label=lab)
ax[1].set_yscale("log")
ax[1].set_xlabel("CMA-ES iteration")
ax[1].set_title("held-out reward terms of the mean theta", fontsize=9)
ax[1].legend(fontsize=8)
for a_ in ax:
    a_.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "L41_cma.png", dpi=110)

show = [1, 2, 3, 4]
ks = np.arange(1, T + 1) * 0.002
fig, ax = plt.subplots(len(show), 1, figsize=(11, 11), sharex=True)
for r, k in enumerate(show):
    _, s0, goal, trs, wm = rows[k]
    for j in range(6):
        line, = ax[r].plot(ks, trs["L4.1 learned"][:, j], lw=1.6)
        c = line.get_color()
        ax[r].axhline(goal[j], color=c, ls="--", lw=0.8)
        ax[r].plot(ks[:len(wm)], wm[:, j], ":", color=c, lw=1.6)
        ax[r].plot(ks, trs["direct goal"][:, j], color=c, lw=0.6, alpha=0.5)
    ax[r].axvline(0.4, color="gray", lw=0.6)
    ax[r].set_title(f"goal #{k}: solid = real MuJoCo (L4.1), dotted = world model (L4.1, 0.4 s), thin = direct goal, "
                    f"dashed = goal", fontsize=9)
    ax[r].grid(alpha=0.3)
    ax[r].set_ylabel("q (rad)")
ax[-1].set_xlabel("time (s)")
fig.tight_layout()
fig.savefig(OUT / "L41_tracking.png", dpi=110)
(OUT / "L41_eval.txt").write_text("\n".join(lines) + "\n")
