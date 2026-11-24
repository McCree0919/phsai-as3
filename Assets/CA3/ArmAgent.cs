using Mujoco;
using UnityEngine;

public class ArmAgent : MonoBehaviour
{
    public int JointCount;
    public int StateSize;
    public int ActionSize;

    public MjActuator gripper;
    public MjBody endEffectorBody;
    public Transform endEffector;
    public MjHingeJoint[] joints;
    public MjActuator[] actuators;

    float[] action;
    float[] state;
    bool ready;

    public float[] CurrentAction => action;
    public bool Ready => ready;

    void Awake()
    {
        InitializeJointsAndActuators();
    }

    void FixedUpdate()
    {
        if (ready)
            ApplyAction();
    }

    // Lite 6 (see ROBOT_CONFIG.md): joint1..joint6 driven by position actuators act1..act6.
    public void InitializeJointsAndActuators()
    {
        JointCount = 6;
        ActionSize = 6;
        StateSize = 2 * JointCount; // [q(6), qd(6)]

        joints = new MjHingeJoint[JointCount];
        actuators = new MjActuator[ActionSize];
        for (int i = 0; i < JointCount; i++)
        {
            joints[i] = FindComponent<MjHingeJoint>("joint" + (i + 1));
            actuators[i] = FindComponent<MjActuator>("act" + (i + 1));
        }
        endEffectorBody = FindComponent<MjBody>("link6");
        endEffector = FindComponent<MjSite>("attachment_site").transform;

        action = new float[ActionSize];
        state = new float[StateSize];
        ready = true;
    }

    // Joint position targets in radians, act1..act6 order.
    public void SetAction(float[] targets)
    {
        System.Array.Copy(targets, action, ActionSize);
    }

    // MjScene copies Control into data->ctrl after each mj_step, so this takes effect on the next step.
    public void ApplyAction()
    {
        for (int i = 0; i < ActionSize; i++)
            actuators[i].Control = action[i];
    }

    // [q1..q6 (rad), qd1..qd6 (rad/s)] read directly from MuJoCo qpos/qvel.
    public float[] Observe()
    {
        unsafe
        {
            var data = MjScene.Instance.Data;
            for (int i = 0; i < JointCount; i++)
            {
                state[i] = (float)data->qpos[joints[i].QposAddress];
                state[JointCount + i] = (float)data->qvel[joints[i].DofAddress];
            }
        }
        return state;
    }

    public void CopyState(float[] destination)
    {
        System.Array.Copy(Observe(), destination, StateSize);
    }

    private T FindComponent<T>(string name) where T : Component
    {
        foreach (var component in GetComponentsInChildren<T>(true))
            if (component.name == name)
                return component;
        Debug.LogError($"ArmAgent: no {typeof(T).Name} named '{name}' under {this.name}.");
        return null;
    }
}
