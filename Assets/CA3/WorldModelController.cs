using System;
using UnityEngine;


[DefaultExecutionOrder(-5)]
public class WorldModelController : MonoBehaviour
{
    public ArmAgent agent;
    public ArmJointController goals;
    public TextAsset controllerJson;

    [Range(0.0f, 1f)]
    public float actionSmoothing = 0.20f; // = train_controller.ACTION_SMOOTH

    public float maxDeltaPerStep = 0.05f;

    // L5: joint goals come from the IK when it is assigned and enabled, otherwise from ArmJointController.
    public IkGoalProvider ik;
    // Fallback for debugging (key T in Play Mode): W = b = 0, i.e. raw = goal, same smoothing and timing.
    public bool thetaZero = false;

    public float[] Goal => ik != null && ik.isActiveAndEnabled ? ik.JointGoal : goals.CurrentTarget;

    float[] W;
    float[] b;
    float[] jointLo;
    float[] jointHi;

    float[] features;
    float[] rawAction;
    float[] action;
    bool ready;
    bool hasAction;
    bool tookOverEmitActions;
    bool previousEmitActions;

    void Start()
    {
        if (agent == null)
            agent = GetComponent<ArmAgent>();
        if (goals == null)
            goals = GetComponent<ArmJointController>();
        if (agent == null || goals == null || controllerJson == null)
        {
            Debug.LogError("WorldModelController needs agent, goals, and controller.json.");
            enabled = false;
            return;
        }

        features = new float[agent.StateSize];
        rawAction = new float[agent.ActionSize];
        action = new float[agent.ActionSize];

        if (!Load(controllerJson.text))
        {
            enabled = false;
            return;
        }

        previousEmitActions = goals.emitActions;
        goals.emitActions = false;
        tookOverEmitActions = true;
        ready = true;
        Debug.Log(
            $"WorldModelController ready (CMA-ES Controller C, {controllerJson.name}). " +
            $"Goals come from {(ik != null && ik.isActiveAndEnabled ? "IkGoalProvider" : "ArmJointController")}; " +
            "actions come from the learned linear policy.");
    }

    void OnEnable()
    {
        if (!ready || goals == null)
            return;

        previousEmitActions = goals.emitActions;
        goals.emitActions = false;
        tookOverEmitActions = true;
        hasAction = false;
    }

    void OnDisable()
    {
        if (tookOverEmitActions && goals != null)
        {
            goals.emitActions = previousEmitActions;
            tookOverEmitActions = false;
        }
        hasAction = false;
    }

    void FixedUpdate()
    {
        if (!ready || !agent.Ready)
            return;

        var state = agent.Observe();
        if (!hasAction)
        {
            // Smoothing starts from the current joint positions (as in train_controller.rollout).
            System.Array.Copy(state, action, agent.ActionSize);
            hasAction = true;
        }
        ActionFrom(state, action);
        agent.SetAction(action);
    }

    // destination holds the previous action on entry (smoothing state) and the new action on exit.
    public void ActionFrom(float[] state, float[] destination)
    {
        ActionFrom(state, Goal, destination);
    }

    // Same as train_controller.policy + smooth_action:
    // raw = goal + W [q - goal, qd] + b, then blend / clamp the change / clamp to joint limits.
    public void ActionFrom(float[] state, float[] goal, float[] destination)
    {
        int n = agent.ActionSize;
        int f = agent.StateSize;
        for (int j = 0; j < f; j++)
            features[j] = j < n ? state[j] - goal[j] : state[j];
        for (int i = 0; i < n; i++)
        {
            float sum = goal[i];
            if (!thetaZero)
            {
                sum += b[i];
                for (int j = 0; j < f; j++)
                    sum += W[i * f + j] * features[j];
            }
            rawAction[i] = sum;
        }
        ApplySmoothing(rawAction, destination);
    }

    void ApplySmoothing(float[] raw, float[] smoothed)
    {
        for (int i = 0; i < raw.Length; i++)
        {
            float prev = smoothed[i];
            float blended = (1f - actionSmoothing) * prev + actionSmoothing * raw[i];
            float delta = Mathf.Clamp(blended - prev, -maxDeltaPerStep, maxDeltaPerStep);
            smoothed[i] = Mathf.Clamp(prev + delta, jointLo[i], jointHi[i]);
        }
    }

    bool Load(string json)
    {
        try
        {
            var data = JsonUtility.FromJson<ControllerFile>(json);
            if (data == null || data.W == null || data.b == null ||
                data.W.Length != agent.ActionSize * agent.StateSize ||
                data.b.Length != agent.ActionSize)
            {
                Debug.LogError(
                    "controller.json has unexpected W/b shapes for ActionSize=" +
                    agent.ActionSize + " StateSize=" + agent.StateSize + ".");
                return false;
            }

            W = data.W;
            b = data.b;
            jointLo = data.joint_lo != null && data.joint_lo.Length == agent.ActionSize
                ? data.joint_lo
                : CopyRangesLo();
            jointHi = data.joint_hi != null && data.joint_hi.Length == agent.ActionSize
                ? data.joint_hi
                : CopyRangesHi();
            return true;
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            return false;
        }
    }

    float[] CopyRangesLo()
    {
        var result = new float[agent.ActionSize];
        for (int i = 0; i < agent.ActionSize; i++)
            result[i] = goals.jointRanges != null && i < goals.jointRanges.Length
                ? goals.jointRanges[i].x
                : -Mathf.PI;
        return result;
    }

    float[] CopyRangesHi()
    {
        var result = new float[agent.ActionSize];
        for (int i = 0; i < agent.ActionSize; i++)
            result[i] = goals.jointRanges != null && i < goals.jointRanges.Length
                ? goals.jointRanges[i].y
                : Mathf.PI;
        return result;
    }

    [Serializable]
    class ControllerFile
    {
        public string feature;
        public int state_size;
        public int action_size;
        public float[] W;
        public float[] b;
        public float[] joint_lo;
        public float[] joint_hi;
        public float[] home;
    }
}
