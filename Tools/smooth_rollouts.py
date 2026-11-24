"""L3 extra check: MuJoCo test trajectories with L4-style smoothed actions (NOT training data).

Needs the `mujoco==3.17.0` Python package (same version as the Unity plugin):
    python Tools/smooth_rollouts.py                     -> TestResults/L3_smooth_rollouts.npz (seed 123, 8 episodes)
    python Tools/smooth_rollouts.py 2027 16 L31_smooth_test_seed2027.npz   (independent L3.1 test set)

1) Sanity: replay recorded actions from rollout.csv in Python MuJoCo and compare with the Unity trajectory.
2) Generate EPISODES (default 8) x 200 steps. The actions follow train_controller.py:
   goal = sample_goal(), action = smooth_action(prev, raw)  (blend 0.2, |delta| <= 0.05 rad/step).
   First half (0-3 of 8): raw = goal (the policy at theta = 0 of a goal-tracking controller), start at home.
   Second half (4-7 of 8): raw = goal + W @ [q - goal, qd] + b, W ~ N(0, 0.1), b ~ N(0, 0.05) (CMA-ES style
   exploration), starting from the pose reached after holding a random target for 1 s.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
XML = PROJECT / "Robots/ufactory_lite6/lite6.xml"
CSV = PROJECT / "Assets/CA3/Data/rollout.csv"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 123
EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 8
OUT = PROJECT / "TestResults" / (sys.argv[3] if len(sys.argv) > 3 else "L3_smooth_rollouts.npz")

# Same constants as train_controller.py (Lite 6 values from ROBOT_CONFIG.md).
HOME = np.array([0, 0, 1.57, 0, 1.57, 0], dtype=np.float32)
JOINT_LO = np.array([-6.28319, -2.61799, -0.061087, -6.28319, -2.1642, -6.28319], dtype=np.float32)
JOINT_HI = np.array([6.28319, 2.61799, 5.23599, 6.28319, 2.1642, 6.28319], dtype=np.float32)
RANDOM_SCALE, ACTION_SMOOTH, MAX_DELTA, HORIZON = 0.35, 0.2, 0.05, 200

model = mujoco.MjModel.from_xml_path(str(XML))
model.opt.timestep = 0.002
data = mujoco.MjData(model)


def state():
    return np.r_[data.qpos[:6], data.qvel[:6]].astype(np.float32)


def sample_goal(rng):
    span = (JOINT_HI - JOINT_LO) * 0.5 * RANDOM_SCALE
    return np.clip(HOME + rng.uniform(-span, span), JOINT_LO, JOINT_HI).astype(np.float32)


def smooth_action(prev, raw):
    blended = (1.0 - ACTION_SMOOTH) * prev + ACTION_SMOOTH * raw
    delta = np.clip(blended - prev, -MAX_DELTA, MAX_DELTA)
    return np.clip(prev + delta, JOINT_LO, JOINT_HI).astype(np.float32)


# ---- 1) Python MuJoCo vs the Unity rollout (same recorded actions) ----
rows = np.loadtxt(CSV, delimiter=",", dtype=np.float32)
worst = 0.0
for i0 in range(56000, 69800, 1400):  # 10 windows of 200 steps in the validation block
    mujoco.mj_resetData(model, data)
    data.qpos[:6], data.qvel[:6] = rows[i0, :6], rows[i0, 6:12]
    for k in range(200):
        data.ctrl[:] = rows[i0 + k, 12:18]
        mujoco.mj_step(model, data)
    worst = max(worst, np.abs(state()[:6] - rows[i0 + 199, 18:24]).max())
print(f"sanity: Python MuJoCo vs Unity rollout, 10 windows x 200 steps, max |q error| at step 200 = {worst:.2e} rad")

# ---- 2) smooth-action test episodes ----
rng = np.random.default_rng(SEED)
S, A, NS, G, KIND = [], [], [], [], []
for ep in range(EPISODES):
    mujoco.mj_resetData(model, data)
    data.qpos[:6] = HOME
    data.ctrl[:] = HOME
    if ep >= EPISODES // 2:  # start from a random settled-ish pose
        data.ctrl[:] = sample_goal(rng)
        for _ in range(500):
            mujoco.mj_step(model, data)
    goal = sample_goal(rng)
    W = rng.normal(0, 0.1, (6, 12)).astype(np.float32) if ep >= EPISODES // 2 else np.zeros((6, 12), np.float32)
    b = rng.normal(0, 0.05, 6).astype(np.float32) if ep >= EPISODES // 2 else np.zeros(6, np.float32)
    prev = state()[:6].copy()
    s_ep, a_ep, ns_ep = [], [], []
    for k in range(HORIZON):
        s = state()
        raw = goal + W @ np.r_[s[:6] - goal, s[6:]] + b
        a = smooth_action(prev, raw)
        prev = a
        data.ctrl[:] = a
        mujoco.mj_step(model, data)
        s_ep.append(s); a_ep.append(a); ns_ep.append(state())
    S.append(s_ep); A.append(a_ep); NS.append(ns_ep); G.append(goal)
    KIND.append("raw=goal, start home" if ep < EPISODES // 2 else "random linear policy, random start")
np.savez(OUT, state=np.array(S), action=np.array(A), next_state=np.array(NS), goal=np.array(G),
         kind=np.array(KIND), sanity_max_q_err=worst)
print(f"saved {OUT}: state {np.array(S).shape}, action {np.array(A).shape}")
