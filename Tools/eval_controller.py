"""L4 validation of the learned controller (controller.json) in real MuJoCo 3.17.0 (Python, same lite6.xml).

python Tools/eval_controller.py   -> TestResults/L4_eval.txt, L4_cma.png, L4_tracking.png
1) Unity consistency: re-compute WorldModelController's actions from the Unity log (L4ControllerTest)
   and re-simulate the same closed loop in Python MuJoCo.
2) Goal tracking from rest: 5-goal smoke test (seed 7) and 20-goal test (seed 2026); training used seeds 0-59
   and 1000 (held-out log). Learned controller vs theta = 0 (smoothed goal) vs direct goal (ctrl = goal).
   "Unity timing": the action computed from the state before step t is used by mj_step t+2 (measured in L2/L4).
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
ctrl_file = json.loads((tc.ROOT / "controller.json").read_text())
W = np.array(ctrl_file["W"], dtype=np.float32).reshape(tc.ACTION, tc.FEAT)
B = np.array(ctrl_file["b"], dtype=np.float32)
mjm = mujoco.MjModel.from_xml_path(str(PROJECT / "Robots/ufactory_lite6/lite6.xml"))
mjm.opt.timestep = 0.002
data = mujoco.MjData(mjm)
T = 1000          # 2 s real test (training horizon is the first 200 steps)
DELAY = 2         # Unity: SetAction in FixedUpdate t reaches data->ctrl for mj_step t+2
lines = []


def out(text=""):
    print(text)
    lines.append(text)


def state():
    return np.r_[data.qpos[:6], data.qvel[:6]].astype(np.float32)


def action(mode, s, goal, prev):
    if mode == "direct":
        return goal.copy()
    raw = goal if mode == "theta0" else tc.policy(s, goal, W, B)
    return tc.smooth_action(prev, raw)


def rmse(e, axis=None):
    return np.sqrt((e ** 2).mean(axis))


# ---- 1) Unity consistency ----
log = np.loadtxt(OUT / "L4_unity_controller.csv", delimiter=",", dtype=np.float32)
us, ug, uc = log[:, :12], log[:, 12:18], log[:, 18:24]
out("1) Unity consistency (1500 Unity steps, goals: home then random every 1 s)")
e = [np.abs(tc.smooth_action(uc[t + 1], tc.policy(us[t], ug[t], W, B)) - uc[t + 2]).max() for t in range(len(log) - 2)]
out(f"   ctrl[t+2] == smooth_action(ctrl[t+1], policy(state_t, goal_t)) for all t: max |diff| = {max(e):.1e}")
# WorldModelController's first FixedUpdate runs one frame before the first logged step (ctrl[1] is its first action).
prev = action("learned", us[0], ug[0], us[0, :6].copy())
acts = [prev]
for t in range(len(log)):
    prev = action("learned", us[t], ug[t], prev)
    acts.append(prev)
acts = np.array(acts)  # acts[k] = action of frame k - 1
for d in (1, 2, 3):
    out(f"   ctrl[t] == Python action of frame t-{d}: max |diff| = {np.abs(uc[d:] - acts[1:len(log) + 1 - d]).max():.2e}")
mujoco.mj_resetData(mjm, data)
data.qpos[:6], data.qvel[:6] = us[0, :6], us[0, 6:]
queue = [uc[0], acts[0]]
prev, sim = acts[0].copy(), []
for t in range(len(log)):
    s = state()
    sim.append(s)
    prev = action("learned", s, ug[t], prev)
    queue.append(prev)
    data.ctrl[:] = queue.pop(0)
    mujoco.mj_step(mjm, data)
sim = np.array(sim)
out(f"   Python MuJoCo closed loop (Unity timing) vs Unity: max |q diff| over 1500 steps = {np.abs(sim[:, :6] - us[:, :6]).max():.2e} rad")


# ---- 2) goal tracking ----
def settle(start):
    mujoco.mj_resetData(mjm, data)
    data.qpos[:6] = start
    data.ctrl[:] = start
    for _ in range(500):
        mujoco.mj_step(mjm, data)
    return state()


def run_real(mode, start, goal, delay):
    s0 = settle(start)
    queue = [start.copy()] * delay
    prev, traj, acts = s0[:6].copy(), [], []
    for _ in range(T):
        prev = action(mode, state(), goal, prev)
        acts.append(prev)
        queue.append(prev)
        data.ctrl[:] = queue.pop(0)
        mujoco.mj_step(mjm, data)
        traj.append(state())
    return s0, np.array(traj), np.array(acts), data.ncon


def run_wm(mode, s0, goal, steps=tc.HORIZON):
    st, prev, traj, acts = s0.copy(), s0[:6].copy(), [], []
    for _ in range(steps):
        prev = action(mode, st, goal, prev)
        acts.append(prev)
        st = tc.step_dynamics(model, st, prev, *stats)
        traj.append(st)
    return np.array(traj), np.array(acts)


def metrics(s0, traj, goal):
    q = traj[:, :6]
    d = goal - s0[:6]
    mv = np.abs(d) > 0.05
    over = np.where(mv, np.clip(((q - goal) * np.sign(d)).max(0), 0, None), 0)
    p2p = (q[-250:].max(0) - q[-250:].min(0)).max()  # oscillation amplitude in the last 0.5 s
    return dict(e200=rmse(q[199] - goal), e1000=rmse(q[-1] - goal), emax=np.abs(q[-1] - goal).max(),
                over=over.max(), p2p=p2p, qd_end=np.abs(traj[-100:, 6:]).max())


def reward_terms(traj, actions, start, goal):
    # The teacher's return over the first HORIZON steps, split into its three terms (per-step means).
    q, qd = traj[:tc.HORIZON, :6], traj[:tc.HORIZON, 6:]
    a = np.vstack([start, actions[:tc.HORIZON]])
    pos = ((q - goal) ** 2).sum(1).mean()
    vel = tc.VEL_PENALTY * (qd ** 2).sum(1).mean()
    act = tc.ACTION_SMOOTH_PENALTY * (np.diff(a, axis=0) ** 2).sum(1).mean()
    return np.array([pos, vel, act, pos + vel + act])


def test(seed, n, detail):
    rng = np.random.default_rng(seed)
    cases = [(tc.sample_goal(rng), tc.sample_goal(rng)) for _ in range(n)]
    res = {m: [] for m in ("learned", "learned ideal", "theta0", "direct")}
    terms = {k: [] for k in ("learned WM", "learned real", "learned real no delay", "theta0 WM", "theta0 real")}
    rows = []
    for k, (start, goal) in enumerate(cases):
        s0, tr, a_l, ncon = run_real("learned", start, goal, DELAY)
        _, tr_i, a_i, _ = run_real("learned", start, goal, 0)
        _, tr_0, a_0, _ = run_real("theta0", start, goal, DELAY)
        _, tr_d, _, _ = run_real("direct", start, goal, DELAY)
        wm, a_wm = run_wm("learned", s0, goal)
        wm0, a_wm0 = run_wm("theta0", s0, goal)
        for m, t_ in (("learned", tr), ("learned ideal", tr_i), ("theta0", tr_0), ("direct", tr_d)):
            res[m].append(metrics(s0, t_, goal))
        for name, (t_, a_) in (("learned WM", (wm, a_wm)), ("learned real", (tr, a_l)),
                               ("learned real no delay", (tr_i, a_i)), ("theta0 WM", (wm0, a_wm0)),
                               ("theta0 real", (tr_0, a_0))):
            terms[name].append(reward_terms(t_, a_, s0[:6], goal))
        r = res["learned"][-1]
        r.update(wm_vs_real=rmse(wm[-1, :6] - tr[199, :6]), wm_dist=rmse(wm[-1, :6] - goal), ncon=ncon)
        rows.append((k, s0, goal, tr, tr_d, tr_0, wm))
        if detail:
            out(f"   #{k:2d} q0   [{' '.join(f'{v:+.2f}' for v in s0[:6])}]  goal [{' '.join(f'{v:+.2f}' for v in goal)}]")
            out(f"       q@2s [{' '.join(f'{v:+.2f}' for v in tr[-1, :6])}]  err@0.4s {r['e200']:.3f} err@2s {r['e1000']:.3f} "
                f"(max joint {r['emax']:.3f}) overshoot {r['over']:.3f} osc p2p {r['p2p']:.3f} |qd|end {r['qd_end']:.2f} "
                f"contacts {ncon} | direct err@2s {res['direct'][-1]['e1000']:.3f} | WM end vs real@0.4s {r['wm_vs_real']:.3f}"
                f" (WM dist {r['wm_dist']:.3f} vs real {r['e200']:.3f})")
    return res, rows, {k: np.array(v) for k, v in terms.items()}


def summary(res):
    out(f"   {'controller':24s} {'err@0.4s':>9s} {'err@2s':>8s} {'max jnt@2s':>10s} {'overshoot':>10s} {'osc p2p':>8s}"
        f" {'|qd|end':>8s}  (mean; median err@2s)")
    for m, label in (("learned", "learned (Unity timing)"), ("learned ideal", "learned (no delay)"),
                     ("theta0", "theta=0 smoothed goal"), ("direct", "direct goal (ctrl=goal)")):
        r = res[m]
        g = lambda k: np.array([x[k] for x in r])
        out(f"   {label:24s} {g('e200').mean():9.3f} {g('e1000').mean():8.3f} {g('emax').mean():10.3f} {g('over').mean():10.3f}"
            f" {g('p2p').mean():8.3f} {g('qd_end').mean():8.2f}  ; {np.median(g('e1000')):.3f}")


out("\n2a) Smoke test: 5 goals (seed 7), rest start, 2 s")
res5, _, _ = test(7, 5, False)
summary(res5)
out("\n2b) Test: 20 goals (seed 2026), rest start, 2 s. Errors = RMSE over joints of q - goal (rad);"
    " overshoot = max rad past the goal; osc p2p = max joint peak-to-peak in the last 0.5 s")
res, rows, terms = test(2026, 20, True)
out("")
summary(res)
L = res["learned"]
wm_vs = np.array([x["wm_vs_real"] for x in L])
wm_d = np.array([x["wm_dist"] for x in L])
real_d = np.array([x["e200"] for x in L])
out(f"   World model vs real at step 200 (learned, Unity timing): end q RMSE mean {wm_vs.mean():.3f} max {wm_vs.max():.3f};"
    f" predicted dist-to-goal {wm_d.mean():.3f} vs real {real_d.mean():.3f}; "
    f"WM says <0.1 but real >0.2: {int(((wm_d < 0.1) & (real_d > 0.2)).sum())}/20")
out("   Teacher reward terms over the first 200 steps (per-step means; fitness = -return = total):")
out(f"   {'':24s} {'|q-goal|^2':>11s} {'0.05|qd|^2':>11s} {'0.1|da|^2':>10s} {'total':>8s}")
for name, v in terms.items():
    m = v.mean(0)
    out(f"   {name:24s} {m[0]:11.4f} {m[1]:11.4f} {m[2]:10.5f} {m[3]:8.4f}")
blocked = [k for k, x in enumerate(res["direct"]) if x["e1000"] > 0.05]
out(f"   goals the direct baseline does not reach within 0.05 rad in 2 s: {blocked}")

# ---- plots ----
cl = (tc.ROOT / "controller_log.txt").read_text().splitlines()
cl = np.array([l.split(",") for l in cl[2:]], dtype=float)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(cl[:, 0], cl[:, 1], label="best fitness (iteration's episodes)")
ax.plot(cl[:, 0], cl[:, 2], label="median fitness")
ax.plot(cl[:, 0], cl[:, 3], label="mean theta, held-out episodes")
ax.set_xlabel("CMA-ES iteration")
ax.set_ylabel("fitness = -imagined return (lower is better)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "L4_cma.png", dpi=110)

show = [0, 1, 2, 3]
ks = np.arange(1, T + 1) * 0.002
fig, ax = plt.subplots(len(show), 1, figsize=(11, 11), sharex=True)
for r, k in enumerate(show):
    _, s0, goal, tr, tr_d, _, wm = rows[k]
    for j in range(6):
        line, = ax[r].plot(ks, tr[:, j], lw=1.6)
        c = line.get_color()
        ax[r].axhline(goal[j], color=c, ls="--", lw=0.8)
        ax[r].plot(ks[:len(wm)], wm[:, j], ":", color=c, lw=1.6)
        ax[r].plot(ks, tr_d[:, j], color=c, lw=0.6, alpha=0.5)
    ax[r].axvline(0.4, color="gray", lw=0.6)
    ax[r].set_title(f"goal #{k}: solid = real MuJoCo (learned), dotted = world model (learned, 0.4 s), "
                    f"thin = direct goal, dashed = goal", fontsize=9)
    ax[r].grid(alpha=0.3)
    ax[r].set_ylabel("q (rad)")
ax[-1].set_xlabel("time (s)")
fig.tight_layout()
fig.savefig(OUT / "L4_tracking.png", dpi=110)
(OUT / "L4_eval.txt").write_text("\n".join(lines) + "\n")
