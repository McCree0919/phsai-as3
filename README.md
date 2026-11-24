# CA3 | UFactory Lite 6 World Model

This project tracks a movable target with a six-joint UFactory Lite 6 in Unity and MuJoCo. A damped least-squares IK solver turns the target position into joint goals. The L4.2 controller sends joint position commands to MuJoCo, and a learned dynamics model draws a ghost arm and forecasts its end-effector path.

## Run the demo

1. Open the project on Windows with Unity `6000.3.24f1`. The project includes the MuJoCo `3.17.0` plug-in and its Windows x64 native library. Unity resolves Inference Engine `2.6.1` from the project manifest.
2. Open [CA3.unity](Assets/Scenes/CA3.unity) and enter Play Mode. The saved scene already loads `controller_l42.json`, `dynamics_multistep.onnx`, and `norm_multistep.json` from [Assets/CA3/Data](Assets/CA3/Data).
3. Focus the Game view. Use **W/A/S/D** to move the red target horizontally and **Q/E** to lower or raise it. You can also move `TrackingTarget` with the Scene view Move tool. The green marker shows the live end effector.
4. Press **T** to compare the learned policy with the smoothed direct-goal baseline. The overlay shows the active controller and both IK and live tracking errors.

## Recheck the saved results

Unity Play Mode uses the shipped model and controller. The Python scripts require NumPy, PyTorch, CMA, MuJoCo `3.17.0`, Matplotlib, ONNX, and ONNX Runtime. Run these checks from the project root.

```powershell
python -B Tools/check_rollout.py
python -B Tools/replay_check.py rollout_smooth.csv
python -B Tools/check_ctrl_timing.py L42_unity_controller.csv controller_l42.json
```

The Unity editor checks are [L4ControllerTest.Run](Assets/Tests/Editor/L4ControllerTest.cs) with `-controller controller_l42.json` and [L5Test.Run](Assets/Tests/Editor/L5Test.cs). Pass each method to Unity with `-batchmode -projectPath <project-root> -executeMethod <method>`. Keep graphics enabled for L5 screenshots. Run the checks from a project copy to preserve the original files in [TestResults](TestResults).

The [report](Assets/CA3/REPORT.md) explains the training and evaluation, and the [reflection](CA3_REFLECTION_FINAL.md) answers the six course prompts. [Robot configuration](ROBOT_CONFIG.md) records the joint limits, home pose, units, and sources. The original course starter archive is [CA3_Scripts-1.zip](CA3_Scripts-1.zip).
