# CA3 Report

**Yingchao Cai** · CS6983 · Coding Assignment 3

**Environment** · Windows, Unity `6000.3.24f1`, MuJoCo Unity plug-in `3.17.0`, and Unity Inference Engine `2.6.1`.

## Why the Lite 6

I chose the no-gripper UFACTORY Lite 6 because its six position actuators and end-effector site fit the assignment. That let me focus on recording joint motion, learning the dynamics, and implementing IK. [MuJoCo Menagerie gallery](https://github.com/google-deepmind/mujoco_menagerie) · [Robot configuration and provenance](../../ROBOT_CONFIG.md)

## Pipeline and rollout data

I imported the Lite 6 into Unity and used six joint angles and six velocities as the state. The recorder saves 30 values per MuJoCo step of 0.002 s, including the current state, six actuator targets, and the next state. It reads `data->ctrl` immediately before `mj_step` to capture the action MuJoCo applies on that step. I trained a world model to predict changes in joint state and used it to train a controller with CMA-ES, then supplied the controller's joint goals through IK in Unity.

The step-target rollout contains 70,000 rows, using seed 0 and a one-second goal hold. The 50,000-row smooth rollout uses seed 1 and holds goals for 0.4 s. Its targets alternate between smoothed direct goals and smoothed goals with a random linear correction. The step-target rollout passes the original alignment check. On the smooth rollout the regression heuristic flags alignment, while a one-step MuJoCo replay gives a median maximum state error of `6.6e-7` for the recorded action versus `0.68` and `0.21` for the previous and next actions. [Step-target check](../../TestResults/L2_rollout_check.txt) · [Smooth-rollout check](../../TestResults/L31_rollout_check.txt) · [Replay](../../TestResults/L31_replay_check.txt)

## World model results

The arm has six degrees of freedom (`ACTION = 6`, `STATE = 12`). The MLP has layers `18 → 256 → 256 → 12` with SiLU hidden activations. It takes normalized `[q, qd, action]` and predicts a normalized change in state. Training used Adam, a cosine schedule, 15 epochs, batch size 256, and an initial learning rate of 0.001. The loss is the sum of one-step MSE and 20-step rollout MSE, with each window staying inside a single trajectory. The first 80% of each rollout supplied training data and normalization statistics, and the last 20% supplied validation data (rows `[56000, 70000)` and `[40000, 50000)`). The final checkpoint was epoch 7, which had the lowest mean 20-step error ratio against persistence across the two validation blocks (`0.209449`).

The table uses the training logs' joint-position metric, `sqrt(mean(sum((q_pred - q_true)^2)))`, with the sum over six joints and the mean over starting points. Open-loop predictions start from a recorded state, apply the recorded actions, and feed each prediction into the next step. The evaluation subsamples evenly spaced starts, targeting roughly 2,000 per horizon. Dividing the final model's 100-step result on step targets by `sqrt(6)` gives an aggregate per-joint RMSE of about `0.0902` rad.

| Checkpoint and validation set | One step | 10 steps | 50 steps | 100 steps |
| --- | ---: | ---: | ---: | ---: |
| L3 baseline on step targets | 0.001123 | 0.016335 | 0.129074 | 0.224450 |
| Final model on step targets | 0.000962 | 0.016675 | 0.137099 | 0.221034 |
| Final model on smooth actions | 0.002526 | 0.050856 | 0.460653 | 0.770068 |

For the independent seed-2027 smooth test with random policy corrections, the 100-step per-joint RMSE dropped from `0.391` rad for L3 to `0.260` rad for the final model. With the direct-goal action sequence, the final model improved at 100 steps and had more error than L3 at 200 steps. Error still grows in open loop, particularly on smooth actions. [L3 training log](Data/train_log.txt) · [Final training log](Data/train_log_multistep.txt) · [Independent evaluation](../../TestResults/L31ms_eval.txt)

## Controller results

L4.2 uses a 12-parameter per-joint correction, `raw[j] = goal[j] + kp[j] * (q[j] - goal[j]) + kd[j] * qd[j]`. It blends the new target into the previous command by 0.2, limits each change to 0.05 rad, and clamps to the joint limits. Training models the two-step delay measured in Unity. Its cost per step is the sum of squared position error, squared velocity weighted by 0.005, and squared changes in the smoothed command weighted by 0.1.

CMA-ES ran for 60 iterations with population 11, initial scale 0.1, and 16 imagined episodes of 200 steps per candidate. The training episode seeds ran from 0 to 59. On fixed validation episodes (seed 1000), mean return was `−3.61637` for the zero-correction baseline, `−3.39183` for the best candidate, and `−3.38948` for the final population mean. The final mean's logged cost was `3.38948`, with `2.92021` from position error, `0.46914` from velocity, and `0.00014` from action changes. Its mean imagined return sums to about `−677.9` per 200-step episode. The final mean was exported to `controller_l42.json`. [Training and model selection](Data/controller_l42_log.txt) · [Unity action timing](../../TestResults/L42_timing_check.txt)

The 1,500-step Unity controller check used the same home goal and first two random joint goals as the seed-0 collection rollout. The goal values match in order in the Unity controller trace and the collection CSV. [Unity controller trace](../../TestResults/L42_unity_controller.csv) · [Collection rollout](Data/rollout.csv)

The separate MuJoCo test held 20 sampled joint goals for two seconds each. Across the 16 reachable goals, L4.2 averaged `0.097` rad joint RMSE at 0.4 s, compared with `0.260` for L4.1 and `0.128` for both direct-goal baselines. At two seconds, errors were `0.012` rad for L4.2, `0.011` rad for direct goal, and `0.010` rad for smoothed direct goal (`θ = 0`). The four remaining goals were blocked by self-contact in the baseline reachability check. [Joint-goal evaluation](../../TestResults/L42_eval.txt) · [Tracking plot](../../TestResults/L42_tracking.png)

## WorldModelDriver

The CPU inference worker loads `dynamics_multistep.onnx` (opset 15, input `[1,18]`, output `[1,12]`) and `norm_multistep.json`. The driver normalizes the input outside the model, de-normalizes the predicted delta, and adds it back to the current state. It uses its own copy of the controller's two-step action queue for the ghost arm. The scene blends 5% of the measured state into the ghost at each step and reanchors the ghost and action history on startup, after 3,000 steps, or when joint RMSE exceeds 0.2 rad. Every 50 steps the driver predicts the next 200 steps (0.4 s) using the current joint goal and draws the end-effector path.

In 140 sampled predictions, the largest difference between Unity and PyTorch was `1.07e-6` rad in position and `4.77e-6` rad/s in velocity. This comparison checks that both implementations produce the same output for the same input. The driver-on and driver-off runs logged the same joint angles and commands across 7,000 physics steps. [Unity prediction and driver comparison](../../TestResults/L5_eval.txt)

## IK and live tracking

The solver uses the `attachment_site` position Jacobian and damped least squares with damping 0.05. A null-space term pulls the pose toward home. The solver uses separate MuJoCo data and starts from the previous solution. If the first solve misses the 0.5 mm tolerance, it tries again from home and switches to the second solution if the error improves by more than 1 mm. Each solve uses up to 30 iterations in the live scene, with each joint update capped at 0.2 rad per iteration. Live IK limits joint 5 to ±1.6 rad to avoid a known self-contact posture. The solver handles position and joint limits. Collision-free paths for arbitrary targets would require a separate planner.

All eight reachable FK targets in the IK-only test converged within 0.38 mm. In the Unity closed-loop test, the six reachable targets ended 0.55–2.49 mm from the live end effector and reached the 5 mm band in 0.42–0.56 s. The unreachable target at `(0.900, 0.500, 0.300)` m triggered non-convergence and stayed about 477 mm away after two seconds. The contact checks sampled every 50th physics step and found no contact in these runs. `TrackingTarget` has no Collider and can be moved with the Scene view Move tool during Play Mode. The automated test moves its Transform directly. [Demo instructions](../../README.md) · [IK test](../../TestResults/L5_ik.txt) · [Unity evaluation](../../TestResults/L5_eval.txt) · [Live tracking plot](../../TestResults/L5_loop.png)

## Design question

Rollout collection can stay in joint space because the dynamics model learns how joint positions and velocities respond to actuator targets. A Cartesian goal leaves those dynamics the same. At runtime, IK converts the target's XYZ position into a joint-angle goal, and the same learned controller follows that goal. The target can move around the workspace using the existing controller.

The [six reflection responses](../../CA3_REFLECTION_FINAL.md) accompany this report.

## AI assistance

I used AI tools to help design the model training pipeline, implement parts of the code, and debug issues during development. AI also assisted with testing and organizing the report. I reviewed the results and verified the final implementation in Unity.
