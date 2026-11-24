// L2 batchmode rollout collection.
// Setup:   Unity.exe -batchmode -quit -nographics -projectPath . -executeMethod L2RolloutTest.Setup
// Collect: Unity.exe -batchmode -nographics -projectPath . -executeMethod L2RolloutTest.Collect -steps 70000
// Output:  Assets/CA3/Data/rollout.csv, log summary in TestResults/L2_collect.txt
using System.IO;
using System.Linq;
using Mujoco;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

[InitializeOnLoad]
public static class L2RolloutTest
{
    const string ScenePath = "Assets/Scenes/CA3.unity";
    const string Flag = "CA3_L2_COLLECT";
    const string RolloutPath = "Assets/CA3/Data/rollout.csv";

    static bool hooked, done;
    static float[] lastTarget;
    static int sinceSwitch = -1;
    static readonly System.Collections.Generic.Dictionary<int, int> delays = new();

    static L2RolloutTest()
    {
        if (SessionState.GetBool(Flag, false))
            EditorApplication.update += Tick;
    }

    // Scene config for rollout collection (values from ROBOT_CONFIG.md, radians).
    public static void Setup()
    {
        var scene = EditorSceneManager.OpenScene(ScenePath);
        var root = GameObject.Find("lite6");
        var agent = root.GetComponent<ArmAgent>();

        var goals = root.GetComponent<ArmJointController>();
        if (goals == null)
            goals = root.AddComponent<ArmJointController>();
        goals.agent = agent;
        goals.holdSeconds = 1f;
        goals.randomScale = 0.35f;
        goals.emitActions = true;
        goals.seed = 0;
        goals.homeTargets = new[] { 0f, 0f, 1.57f, 0f, 1.57f, 0f };
        goals.jointRanges = new[]
        {
            new Vector2(-6.28319f, 6.28319f),
            new Vector2(-2.61799f, 2.61799f),
            new Vector2(-0.061087f, 5.23599f),
            new Vector2(-6.28319f, 6.28319f),
            new Vector2(-2.1642f, 2.1642f),
            new Vector2(-6.28319f, 6.28319f),
        };

        var recorder = root.GetComponent<RolloutRecorder>();
        if (recorder == null)
            recorder = root.AddComponent<RolloutRecorder>();
        recorder.agent = agent;
        recorder.controller = goals;
        recorder.steps = 70000;
        recorder.enabled = false; // enable (or run Collect) only when recording; Play would overwrite rollout.csv

        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
    }

    // L3.1: -smooth records L4-style smoothed actions to rollout_smooth.csv (seed 1, 0.4 s per target = 200 steps).
    public static void Collect()
    {
        var args = System.Environment.GetCommandLineArgs();
        int i = System.Array.IndexOf(args, "-steps");
        int steps = i >= 0 ? int.Parse(args[i + 1]) : 70000;
        bool smooth = System.Array.IndexOf(args, "-smooth") >= 0;

        EditorSceneManager.OpenScene(ScenePath);
        var recorder = Object.FindAnyObjectByType<RolloutRecorder>(FindObjectsInactive.Include);
        recorder.steps = steps; // in-memory only, not saved to the scene
        recorder.enabled = true;
        // L5 saved the live-tracking components in the scene; collection uses ArmJointController only
        foreach (var c in new Behaviour[] { Object.FindAnyObjectByType<WorldModelController>(),
                     Object.FindAnyObjectByType<IkGoalProvider>(), Object.FindAnyObjectByType<WorldModelDriver>() })
            if (c != null)
                c.enabled = false;
        Object.FindAnyObjectByType<ArmJointController>().emitActions = true;
        if (smooth)
        {
            var goals = Object.FindAnyObjectByType<ArmJointController>();
            goals.smoothActions = true;
            goals.holdSeconds = 0.4f;
            goals.seed = 1;
            recorder.fileName = "rollout_smooth.csv";
        }
        string path = "Assets/CA3/Data/" + recorder.fileName;
        if (File.Exists(path))
            File.Delete(path);
        SessionState.SetString("CA3_L2_FILE", path);
        SessionState.SetBool("CA3_L2_SMOOTH", smooth);
        SessionState.SetInt("CA3_L2_STEPS", steps);
        SessionState.SetBool(Flag, true);
        EditorApplication.EnterPlaymode();
    }

    static unsafe void Tick()
    {
        if (done || !EditorApplication.isPlaying || !MjScene.InstanceExists || MjScene.Instance.Data == null)
            return;
        bool smooth = SessionState.GetBool("CA3_L2_SMOOTH", false);
        if (!hooked)
        {
            if (!smooth) // the goal == ctrl delay test only applies to step targets
                MjScene.Instance.preUpdateEvent += MeasureDelay;
            hooked = true;
        }
        int steps = SessionState.GetInt("CA3_L2_STEPS", 70000);
        string path = SessionState.GetString("CA3_L2_FILE", RolloutPath);
        if (!File.Exists(path))
            return;

        done = true;
        var summary =
            $"{(smooth ? "L3.1 smoothed-action" : "L2")} rollout collection: requested steps={steps}, sim time={MjScene.Instance.Data->time:F3}s, " +
            $"wall time since startup={EditorApplication.timeSinceStartup:F1}s\n" +
            $"file={path} ({new FileInfo(path).Length / 1024} KB)\n" +
            (smooth ? "" : "goal switch -> data->ctrl delay (physics steps: count) " +
            string.Join(", ", delays.OrderBy(p => p.Key).Select(p => $"{p.Key}: {p.Value}")) + "\n");
        Debug.Log("[L2] " + summary);
        Directory.CreateDirectory("TestResults");
        File.WriteAllText(smooth ? "TestResults/L31_collect.txt" : "TestResults/L2_collect.txt", summary);
        SessionState.EraseBool(Flag);
        EditorApplication.update -= Tick;
        EditorApplication.Exit(0);
    }

    // At each pre-step: how many steps after a goal switch does data->ctrl equal the new goal?
    static unsafe void MeasureDelay(object sender, MjStepArgs e)
    {
        var goals = Object.FindAnyObjectByType<ArmJointController>();
        var agent = goals.agent;
        var target = goals.CurrentTarget;
        if (lastTarget != null && !target.SequenceEqual(lastTarget))
            sinceSwitch = 0;
        lastTarget = (float[])target.Clone();
        if (sinceSwitch < 0)
            return;
        bool applied = Enumerable.Range(0, 6).All(i => (float)e.data->ctrl[agent.actuators[i].MujocoId] == target[i]);
        if (applied)
        {
            delays[sinceSwitch] = delays.TryGetValue(sinceSwitch, out var c) ? c + 1 : 1;
            sinceSwitch = -1;
        }
        else
        {
            sinceSwitch++;
        }
    }
}
