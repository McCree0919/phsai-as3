import json
from pathlib import Path
import time
import cma
import numpy as np
import torch

from train import build_model

ROOT = Path(__file__).resolve().parent / "Data"
DYNAMICS_TAG = "_multistep"
STATE = 12   # Lite 6: q(6), qd(6)
ACTION = 6
FEAT = STATE  # [q - goal, qd]
# L4.2: per-joint PD correction, raw[j] = goal[j] + kp[j] (q[j] - goal[j]) + kd[j] qd[j]  (12 params, b = 0).
# False = L4 / L4.1 full linear policy, W (6x12) + b (78 params).
PD_ONLY = True
N_PARAMS = 2 * ACTION if PD_ONLY else ACTION * FEAT + ACTION

# Lite 6 values from ROBOT_CONFIG.md (rad).
HOME = np.array([0, 0, 1.57, 0, 1.57, 0], dtype=np.float32)
JOINT_LO = np.array([-6.28319, -2.61799, -0.061087, -6.28319, -2.1642, -6.28319], dtype=np.float32)
JOINT_HI = np.array([6.28319, 2.61799, 5.23599, 6.28319, 2.1642, 6.28319], dtype=np.float32)
RANDOM_SCALE = 0.35

VEL_PENALTY = 0.005  # L4.1 (L4 used 0.05, which made the policy trade tracking for low speed)
ACTION_SMOOTH_PENALTY = 0.1
ACTION_SMOOTH = 0.2
MAX_DELTA = 0.05
HORIZON = 200
EPISODES = 16
CMA_ITERS = 60
SIGMA0 = 0.1       # initial CMA-ES step size around theta = 0 (raw action = goal)
SEED = 0           # CMA-ES seed; iteration i imagines the episodes of rng seed i
EVAL_SEED = 1000   # fixed imagined episodes for checkpoint selection
# Unity timing (measured in L4): the action computed from state_t is used by physics step t + 2,
# i.e. ctrl[t+2] = smooth_action(ctrl[t+1], policy(state_t, goal_t)). L4 used 0.
CTRL_DELAY = 2
OUT_NAME = "controller_l42"  # -> Data/controller_l42.json, controller_l42_log.txt (L4: "controller", L4.1: "controller_l41")


def build_dynamics():
    model = build_model()
    path = ROOT / f"dynamics{DYNAMICS_TAG}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run train.py first.")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_norm():
    path = ROOT / f"norm{DYNAMICS_TAG}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run train.py first.")
    data = np.load(path)
    return data["x_mean"], data["x_std"], data["y_mean"], data["y_std"]


def unpack(theta):
    if not PD_ONLY:
        w = theta[: ACTION * FEAT].reshape(ACTION, FEAT)
        b = theta[ACTION * FEAT :]
        return w, b
    # kp on the diagonal of the q - goal block, kd on the diagonal of the qd block, everything else 0.
    w = np.zeros((ACTION, FEAT), dtype=np.float32)
    w[:, :ACTION] = np.diag(theta[:ACTION])
    w[:, ACTION:] = np.diag(theta[ACTION:])
    return w, np.zeros(ACTION, dtype=np.float32)


def policy(state, goal, w, b):
    # Feature "q_minus_goal_and_qd"; the raw action is the goal plus a linear correction,
    # so theta = 0 is plain goal tracking (through smooth_action).
    feat = np.concatenate([state[:ACTION] - goal, state[ACTION:]])
    return (goal + w @ feat + b).astype(np.float32)


@torch.no_grad()
def step_dynamics(model, state, action, x_mean, x_std, y_mean, y_std):
    # L3 contract: x = ([s, a] - x_mean) / x_std; next = s + delta * y_std + y_mean.
    x = (np.concatenate([state, action]) - x_mean) / x_std
    delta = model(torch.as_tensor(x, dtype=torch.float32)[None])[0].numpy()
    return (state + delta * y_std + y_mean).astype(np.float32)


def sample_goal(rng):
    span = (JOINT_HI - JOINT_LO) * 0.5 * RANDOM_SCALE
    goal = HOME + rng.uniform(-span, span)
    return np.clip(goal, JOINT_LO, JOINT_HI).astype(np.float32)


def smooth_action(prev, raw):
    blended = (1.0 - ACTION_SMOOTH) * prev + ACTION_SMOOTH * raw
    delta = np.clip(blended - prev, -MAX_DELTA, MAX_DELTA)
    return np.clip(prev + delta, JOINT_LO, JOINT_HI).astype(np.float32)


def controller_step(state, goal, w, b, prev, queue):
    # One frame of WorldModelController: new smoothed action from the current state; the physics step
    # uses the action from CTRL_DELAY frames earlier (queue holds the CTRL_DELAY pending actions).
    action = smooth_action(prev, policy(state, goal, w, b))
    queue.append(action)
    return action, queue.pop(0)


def rollout(model, theta, x_mean, x_std, y_mean, y_std, rng, record=False, terms=False):
    # One imagined episode: start at rest at a random pose, track a random goal for HORIZON steps.
    # Return = -mean over steps of |q - goal|^2 + VEL_PENALTY |qd|^2 + ACTION_SMOOTH_PENALTY |a - prev|^2.
    w, b = unpack(theta)
    start = sample_goal(rng)
    goal = sample_goal(rng)
    state = np.concatenate([start, np.zeros(ACTION, dtype=np.float32)])
    prev = start.copy()
    queue = [start.copy() for _ in range(CTRL_DELAY)]  # ctrl holds the start pose until the first action arrives
    pos = vel = act = 0.0
    traj = []
    for _ in range(HORIZON):
        action, applied = controller_step(state, goal, w, b, prev, queue)
        act += ACTION_SMOOTH_PENALTY * float(np.sum((action - prev) ** 2))
        prev = action
        state = step_dynamics(model, state, applied, x_mean, x_std, y_mean, y_std)
        pos += float(np.sum((state[:ACTION] - goal) ** 2))
        vel += VEL_PENALTY * float(np.sum(state[ACTION:] ** 2))
        traj.append(state)
    total = -(pos + vel + act)
    if terms:
        return np.array([pos, vel, act]) / HORIZON
    if record:
        return total / HORIZON, start, goal, np.array(traj)
    return total / HORIZON


def fitness(theta, model, x_mean, x_std, y_mean, y_std, seed):
    rng = np.random.default_rng(seed)
    returns = []
    for _ in range(EPISODES):
        returns.append(rollout(model, theta, x_mean, x_std, y_mean, y_std, rng))

    return -float(np.mean(returns))


def fitness_terms(theta, model, x_mean, x_std, y_mean, y_std, seed):
    # Per-step means of the three penalty terms (their sum is the fitness).
    rng = np.random.default_rng(seed)
    return np.mean([rollout(model, theta, x_mean, x_std, y_mean, y_std, rng, terms=True) for _ in range(EPISODES)], 0)


def save_controller(theta, selected=""):
    w, b = unpack(theta)
    payload = {
        "selected": selected,
        "kp": np.asarray(theta[:ACTION], dtype=float).tolist() if PD_ONLY else None,
        "kd": np.asarray(theta[ACTION:], dtype=float).tolist() if PD_ONLY else None,
        "feature": "q_minus_goal_and_qd",
        "state_size": STATE,
        "action_size": ACTION,
        "W": w.astype(float).reshape(-1).tolist(),
        "b": b.astype(float).tolist(),
        "joint_lo": JOINT_LO.astype(float).tolist(),
        "joint_hi": JOINT_HI.astype(float).tolist(),
        "home": HOME.astype(float).tolist(),
        "vel_penalty": VEL_PENALTY,
        "horizon": HORIZON,
    }
    path = ROOT / f"{OUT_NAME}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def main():
    model = build_dynamics()
    stats = load_norm()
    es = cma.CMAEvolutionStrategy(np.zeros(N_PARAMS), SIGMA0, {"seed": SEED, "verbose": -9})
    zero = fitness_terms(np.zeros(N_PARAMS), model, *stats, EVAL_SEED)
    log = [f"STATE={STATE} ACTION={ACTION} N_PARAMS={N_PARAMS} popsize={es.popsize} sigma0={SIGMA0} "
           f"episodes={EPISODES} horizon={HORIZON} iters={CMA_ITERS} seed={SEED} dynamics=dynamics{DYNAMICS_TAG}.pt "
           f"smooth={ACTION_SMOOTH} max_delta={MAX_DELTA} vel_penalty={VEL_PENALTY} ctrl_delay={CTRL_DELAY}; "
           f"theta=0 held-out fitness={zero.sum():.5f} (pos {zero[0]:.5f} vel {zero[1]:.5f} act {zero[2]:.5f}); "
           f"|b| initial = 0",
           "iter,best_fitness,median_fitness,mean_theta_heldout_fitness,heldout_pos,heldout_vel,heldout_act,sigma,|b|"]
    print(log[0])
    iter_best = []  # best candidate of each iteration (on that iteration's training episodes)
    for it in range(CMA_ITERS):
        thetas = es.ask()
        fits = [fitness(t, model, *stats, it) for t in thetas]  # same episodes for every candidate
        es.tell(thetas, fits)
        iter_best.append(np.array(thetas[int(np.argmin(fits))]))
        held = fitness_terms(es.mean, model, *stats, EVAL_SEED)
        b_norm = np.linalg.norm(unpack(es.mean)[1])
        log.append(f"{it + 1},{min(fits):.5f},{np.median(fits):.5f},{held.sum():.5f},"
                   + ",".join(f"{v:.5f}" for v in held) + f",{es.sigma:.4f},{b_norm:.4f}")
        print(log[-1])
    w, b = unpack(es.mean)
    log.append(f"final mean b = [{' '.join(f'{v:+.4f}' for v in b)}], |b| = {np.linalg.norm(b):.4f}")

    # Selection on the fixed validation episodes (EVAL_SEED), never on the training episodes or the MuJoCo test goals.
    val = [fitness(t, model, *stats, EVAL_SEED) for t in iter_best]
    k = int(np.argmin(val))
    log.append("validation fitness of each iteration's best candidate: " + " ".join(f"{v:.5f}" for v in val))
    options = {"theta=0": np.zeros(N_PARAMS), f"best candidate (iter {k + 1})": iter_best[k], "final mean": np.array(es.mean)}
    scores = {}
    for name, theta in options.items():
        t = fitness_terms(theta, model, *stats, EVAL_SEED)
        scores[name] = t.sum()
        log.append(f"validation {name}: fitness {t.sum():.5f} (pos {t[0]:.5f} vel {t[1]:.5f} act {t[2]:.5f}); "
                   f"theta = [{' '.join(f'{v:+.4f}' for v in theta)}]")
    chosen = min(scores, key=scores.get)
    log.append(f"selected: {chosen}")
    for line in log[CMA_ITERS + 2:]:
        print(line)
    save_controller(options[chosen], chosen)
    (ROOT / f"{OUT_NAME}_log.txt").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    current_time = time.time()
    main()
    end_time = time.time()
    print(f"Training time: {end_time - current_time} seconds")
