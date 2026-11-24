# Lite 6 configuration

The scene uses the no-gripper UFactory Lite 6 from MuJoCo Menagerie at revision `370c439`. The local [MJCF](Robots/ufactory_lite6/lite6.xml) adds names `act1` through `act6` to the six position actuators. The original [model license](Robots/ufactory_lite6/LICENSE) and source meshes are included alongside it. The scene uses meshes and materials under [Assets/Local/MjImports/lite6364/Resources](Assets/Local/MjImports/lite6364/Resources). [Menagerie source MJCF](https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/370c439/ufactory_lite6/lite6.xml)

The scene runs with a fixed timestep of 0.002 s and the `implicitfast` integrator. Each joint is a limited hinge with damping `1` and armature `0.1`. The position actuators have `kp = 2000` and `kv = 200`. The hinge limits appear in degrees in the Unity Inspector. MuJoCo state and actuator targets use radians and radians per second.

| Joint | Actuator | Joint and control limits (rad) | Force limit (N·m) | Home (rad) |
| --- | --- | --- | --- | --- |
| `joint1` | `act1` | −6.28319 to 6.28319 | ±50 | 0 |
| `joint2` | `act2` | −2.61799 to 2.61799 | ±50 | 0 |
| `joint3` | `act3` | −0.061087 to 5.23599 | ±32 | 1.57 |
| `joint4` | `act4` | −6.28319 to 6.28319 | ±32 | 0 |
| `joint5` | `act5` | −2.1642 to 2.1642 | ±32 | 1.57 |
| `joint6` | `act6` | −6.28319 to 6.28319 | ±20 | 0 |

`ArmAgent` finds the joints and actuators by their Unity GameObject names. It reads `[q, qd]` from `qpos` and `qvel` and writes six joint position targets. The `ArmJointController` home pose comes from the MJCF home keyframe. The end-effector body is `link6`, and the position used by IK comes from `attachment_site`. In Unity coordinates the home FK position is about `(0.228, 0.468, 0)` m. MuJoCo `(x, y, z)` corresponds to Unity `(x, z, y)`.

The live IK solver limits joint 5 to ±1.6 rad to stay clear of the observed contact between the `link4` and `link6` collision meshes. The full MJCF range remains in use for joint-space collection and controller evaluation.
