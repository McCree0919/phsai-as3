import numpy as np
import torch
import time
from pathlib import Path
from torch import nn

ROOT = Path(__file__).resolve().parent / "Data"
STATE = 12   # Lite 6: q(6), qd(6)
ACTION = 6   # position targets for act1..act6

HIDDEN = 256
EPOCHS = 15
BATCH = 256
LR = 1e-3
HORIZON = 20          # multi-step loss rollout length (20 x 0.002 s)
MULTI_WEIGHT = 1.0
VAL_FRACTION = 0.2    # last 20% of each rollout file (contiguous in time) is validation

DATA_FILES = ["rollout.csv", "rollout_smooth.csv"]
OUT_TAG = "_multistep"   # L3.1 run: "_smooth" with SELECT_BY = "one_step"
# Checkpoint rule. "one_step": lowest val one-step loss (L3, L3.1).
# "multi_step": lowest mean over validation blocks of open_loop_rmse@HORIZON / persistence@HORIZON.
SELECT_BY = "multi_step"
EVAL_HORIZONS = (HORIZON, 50, 100, 200)  # logged every epoch; only HORIZON is used for selection


def build_model():
    # Normalized [state, action] -> normalized (next_state - state).
    return nn.Sequential(
        nn.Linear(STATE + ACTION, HIDDEN), nn.SiLU(),
        nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
        nn.Linear(HIDDEN, STATE),
    )


def mean_std(z):
    return z.mean(0), z.std(0) + 1e-6


def norm(z, mean, std):
    return (z - mean) / std


def save_norm_json(x_mean, x_std, y_mean, y_std, name="norm.json"):
    arrays = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    lines = ["{"]
    keys = list(arrays)
    for index, key in enumerate(keys):
        values = ", ".join(f"{float(value):.8f}" for value in arrays[key])
        comma = "," if index < len(keys) - 1 else ""
        lines.append(f'  "{key}": [{values}]{comma}')
    lines.append("}")
    (ROOT / name).write_text("\n".join(lines) + "\n")


def load_rollout(name="rollout.csv"):
    path = ROOT / name
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"No rollout at {path}. Enable RolloutRecorder and Play.")
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != STATE + ACTION + STATE:
        raise ValueError(f"Expected {STATE + ACTION + STATE} columns, found {data.shape[1]}.")
    return data


def predict_delta(model, state, action, x_mean, x_std, y_mean, y_std):
    x = np.concatenate([state, action], axis=-1)
    x_n = (x - x_mean) / x_std
    with torch.no_grad():
        delta_n = model(torch.as_tensor(x_n)).numpy()
    return delta_n * y_std + y_mean


def joint_rmse(pred, true):
    err = pred[..., 0:ACTION] - true[..., 0:ACTION]
    return float(np.sqrt((err ** 2).sum(axis=-1).mean()))


def open_loop_rmse(model, data, start, x_mean, x_std, y_mean, y_std, horizon):
    end = len(data) - horizon
    if end <= start:
        return float("nan")
    idx = np.arange(start, end)
    if len(idx) > 2000:
        idx = idx[:: max(1, len(idx) // 2000)]
    state = data[idx, 0:STATE].copy()
    for t in range(horizon):
        action = data[idx + t, STATE:STATE + ACTION]
        state = state + predict_delta(model, state, action, x_mean, x_std, y_mean, y_std)
    return joint_rmse(state, data[idx + horizon, 0:STATE])


def persistence_rmse(data, start, horizon):
    # open_loop_rmse of "the state does not change" on the same windows (scale of each validation block).
    idx = np.arange(start, len(data) - horizon)
    if len(idx) > 2000:
        idx = idx[:: max(1, len(idx) // 2000)]
    return joint_rmse(data[idx, 0:STATE], data[idx + horizon, 0:STATE])


def multi_step_loss(model, data, starts, x_mean, x_std, y_mean, y_std, horizon, batch, loss_fn):
    # Roll the model `horizon` steps from random window starts (windows stay inside one training
    # block) using the recorded actions; compare each predicted state with the recorded next state
    # (scaled by state std).
    start = starts[torch.randint(0, len(starts), (batch,))]
    s_std = x_std[:STATE]
    state = data[start, 0:STATE]
    loss = 0.0
    for t in range(horizon):
        action = data[start + t, STATE:STATE + ACTION]
        x = (torch.cat([state, action], dim=-1) - x_mean) / x_std
        state = state + model(x) * y_std + y_mean
        loss = loss + loss_fn(state / s_std, data[start + t, STATE + ACTION:] / s_std)
    return loss / horizon


def train_world_model(files):
    # files: list of (name, rollout array). Each file is one continuous rollout; its first
    # (1 - VAL_FRACTION) rows train and the rest validate, so every file has validation data.
    torch.manual_seed(0)
    splits = [int(len(data) * (1 - VAL_FRACTION)) for _, data in files]
    train_data = np.concatenate([data[:sp] for (_, data), sp in zip(files, splits)])
    val_data = np.concatenate([data[sp:] for (_, data), sp in zip(files, splits)])
    # Multi-step windows must stay inside one file's training block.
    offsets = np.cumsum([0] + splits[:-1])
    starts = torch.as_tensor(np.concatenate([np.arange(o, o + sp - HORIZON + 1) for o, sp in zip(offsets, splits)]))
    n_train = len(train_data)

    def xy(data):
        s, a, s1 = data[:, 0:STATE], data[:, STATE:STATE + ACTION], data[:, STATE + ACTION:]
        return np.concatenate([s, a], axis=1), s1 - s

    x, y = xy(train_data)
    x_mean, x_std = mean_std(x)  # stats from training rows only
    y_mean, y_std = mean_std(y)
    xn = torch.as_tensor(norm(x, x_mean, x_std))
    yn = torch.as_tensor(norm(y, y_mean, y_std))
    xv, yv = xy(val_data)
    xvn = torch.as_tensor(norm(xv, x_mean, x_std))
    yvn = torch.as_tensor(norm(yv, y_mean, y_std))
    t_data = torch.as_tensor(train_data)
    stats = [torch.as_tensor(v) for v in (x_mean, x_std, y_mean, y_std)]

    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    loss_fn = nn.MSELoss()

    config = ("; ".join(f"{name}: rows={len(data)} train=[0,{sp}) val=[{sp},{len(data)})"
                        for (name, data), sp in zip(files, splits))
              + f"; model={STATE + ACTION}-{HIDDEN}-{HIDDEN}-{STATE} SiLU, epochs={EPOCHS} batch={BATCH} lr={LR} "
              f"(Adam, cosine) multi-step horizon={HORIZON} weight={MULTI_WEIGHT}")
    config += f"; checkpoint selection={SELECT_BY}"
    pers = [persistence_rmse(data, sp, HORIZON) for (_, data), sp in zip(files, splits)]
    log = [config, "epoch,train_one_step,train_multi_step,val_one_step"
           + "".join(f",val_open_loop_rmse_q@{h}[{name}]" for name, _ in files for h in EVAL_HORIZONS)
           + f",select_score(mean open_loop@{HORIZON}/persistence@{HORIZON}; persistence="
           + "/".join(f"{p:.6f}" for p in pers) + ")"]
    print(config)
    best, best_state, best_epoch, candidates = float("inf"), None, 0, {}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        one_sum = multi_sum = 0.0
        batches = 0
        for i in range(0, n_train, BATCH):
            b = perm[i:i + BATCH]
            one = loss_fn(model(xn[b]), yn[b])
            multi = multi_step_loss(model, t_data, starts, *stats, HORIZON, BATCH, loss_fn)
            loss = one + MULTI_WEIGHT * multi
            opt.zero_grad()
            loss.backward()
            opt.step()
            one_sum += one.item()
            multi_sum += multi.item()
            batches += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            val = loss_fn(model(xvn), yvn).item()
        val_ol = [[open_loop_rmse(model, data, sp, x_mean, x_std, y_mean, y_std, h) for h in EVAL_HORIZONS]
                  for (_, data), sp in zip(files, splits)]
        score = float(np.mean([v[0] / p for v, p in zip(val_ol, pers)]))
        line = (f"{epoch},{one_sum / batches:.6f},{multi_sum / batches:.6f},{val:.6f}"
                + "".join(f",{v:.6f}" for block in val_ol for v in block) + f",{score:.6f}")
        log.append(line)
        print(line)
        candidates[epoch] = {k: v.clone() for k, v in model.state_dict().items()}
        crit = score if SELECT_BY == "multi_step" else val
        if crit < best:
            best, best_epoch = crit, epoch
            best_state = candidates[epoch]

    model.load_state_dict(best_state)
    model.eval()
    summary = f"selected epoch {best_epoch} by {SELECT_BY}: {best:.6f}"
    for (name, data), sp in zip(files, splits):
        one_step = open_loop_rmse(model, data, sp, x_mean, x_std, y_mean, y_std, 1)
        ol = {h: open_loop_rmse(model, data, sp, x_mean, x_std, y_mean, y_std, h) for h in (10, 50, 100)}
        summary += (f"\n{name} val joint RMSE (q, rad): one-step={one_step:.6f} "
                    + " ".join(f"open-loop@{h}={v:.6f}" for h, v in ol.items()))
    log.append(summary)
    print(summary)

    names = [f"dynamics{OUT_TAG}.pt", f"norm{OUT_TAG}.npz", f"norm{OUT_TAG}.json",
             f"dynamics{OUT_TAG}.onnx", f"train_log{OUT_TAG}.txt"]
    torch.save(model.state_dict(), ROOT / names[0])
    torch.save(candidates, ROOT / f"dynamics{OUT_TAG}_epochs.pt")  # every epoch's weights, for checking the choice
    np.savez(ROOT / names[1], x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    save_norm_json(x_mean, x_std, y_mean, y_std, names[2])
    torch.onnx.export(model, torch.zeros(1, STATE + ACTION), ROOT / names[3],
                      input_names=["x"], output_names=["delta"], opset_version=15, dynamo=False)
    (ROOT / names[4]).write_text("\n".join(log) + "\n")
    print(f"saved {', '.join(names)} to {ROOT}")


if __name__ == "__main__":
    current_time = time.time()
    train_world_model([(name, load_rollout(name)) for name in DATA_FILES])
    end_time = time.time()
    print(f"Training time: {end_time - current_time} seconds")
