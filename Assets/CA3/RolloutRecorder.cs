using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Mujoco;
using UnityEngine;

// Records (state_t, action_t, state_t+1) for every MuJoCo step and writes Data/rollout.csv
// (columns: 12 state, 6 action, 12 next state; no header) once `steps` samples are collected.
public class RolloutRecorder : MonoBehaviour
{
    public ArmAgent agent;
    public ArmJointController controller;
    public int steps = 70000;
    public string fileName = "rollout.csv"; // in Assets/CA3/Data

    readonly List<float[]> states = new();
    readonly List<float[]> actions = new();
    readonly List<float[]> nextStates = new();

    int n;
    float[] prevS;
    float[] prevA;
    bool saved;

    void Start()
    {
        if (agent == null)
            agent = GetComponent<ArmAgent>();
        if (controller == null)
            controller = GetComponent<ArmJointController>();
        // Runs right before mj_step, so qpos/qvel/ctrl here are exactly what this step integrates.
        MjScene.Instance.preUpdateEvent += OnPreStep;
    }

    void FixedUpdate()
    {
        if (!saved && n > 0 && n % 10000 == 0)
            Debug.Log($"RolloutRecorder: {n}/{steps} samples, tracking error {TrackingError():F3} rad");
    }

    unsafe void OnPreStep(object sender, MjStepArgs args)
    {
        if (saved || !enabled)
            return;

        var s = (float[])agent.Observe().Clone();
        var a = new float[agent.ActionSize];
        for (int i = 0; i < a.Length; i++)
            a[i] = (float)args.data->ctrl[agent.actuators[i].MujocoId];

        // prevA drove the step from prevS to s.
        if (prevS != null)
        {
            states.Add(prevS);
            actions.Add(prevA);
            nextStates.Add(s);
            n++;
        }
        prevS = s;
        prevA = a;

        if (n >= steps)
            Save();
    }

    // RMS of (goal - q) over the joints.
    float TrackingError()
    {
        var s = agent.Observe();
        var goal = controller.CurrentTarget;
        float sum = 0f;
        for (int i = 0; i < agent.ActionSize; i++)
            sum += (goal[i] - s[i]) * (goal[i] - s[i]);
        return Mathf.Sqrt(sum / agent.ActionSize);
    }

    void Save()
    {
        var sb = new StringBuilder();
        for (int t = 0; t < n; t++)
        {
            Append(sb, states[t]);
            sb.Append(',');
            Append(sb, actions[t]);
            sb.Append(',');
            Append(sb, nextStates[t]);
            sb.Append('\n');
        }

        var path = Path.Combine(Application.dataPath, "CA3", "Data", fileName);
        File.WriteAllText(path, sb.ToString());
        saved = true;
        Debug.Log($"RolloutRecorder: saved {n} samples to {path}");
    }

    static void Append(StringBuilder sb, float[] values)
    {
        for (int i = 0; i < values.Length; i++)
        {
            if (i > 0)
                sb.Append(',');
            sb.Append(values[i].ToString("G9", CultureInfo.InvariantCulture));
        }
    }
}
