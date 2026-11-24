// L4 batchmode check: run WorldModelController (controller.json) in Unity Play Mode and log every MuJoCo step.
// Unity.exe -batchmode -nographics -projectPath . -executeMethod L4ControllerTest.Run
// Output: TestResults/L4_unity_controller.csv  (per pre-step: state 12, goal 6, data->ctrl 6)
// L4.1: add -controller controller_l41.json  -> TestResults/L41_unity_controller.csv (L4.2: controller_l42.json -> L42_...)
// The scene is not saved; WorldModelController is added in memory only.
using System.Globalization;
using System.IO;
using System.Text;
using Mujoco;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

[InitializeOnLoad]
public static class L4ControllerTest
{
    const string Flag = "CA3_L4_CONTROLLER";
    const int Steps = 1500; // home goal for 1 s, then random goals (ArmJointController, seed 0, 1 s each)

    static bool hooked, done;
    static readonly StringBuilder sb = new();
    static int n;

    static L4ControllerTest()
    {
        if (SessionState.GetBool(Flag, false))
            EditorApplication.update += Tick;
    }

    public static void Run()
    {
        EditorSceneManager.OpenScene("Assets/Scenes/CA3.unity");
        var root = GameObject.Find("lite6");
        // L5 saved a WorldModelController (+ IK, WorldModelDriver) in the scene: reuse it, goals from ArmJointController
        var wmc = root.GetComponent<WorldModelController>();
        if (wmc == null)
            wmc = root.AddComponent<WorldModelController>();
        wmc.ik = null;
        foreach (var c in new Behaviour[] { root.GetComponent<IkGoalProvider>(), root.GetComponent<WorldModelDriver>() })
            if (c != null)
                c.enabled = false;
        wmc.agent = root.GetComponent<ArmAgent>();
        wmc.goals = root.GetComponent<ArmJointController>();
        var args = System.Environment.GetCommandLineArgs();
        int i = System.Array.IndexOf(args, "-controller");
        string json = i >= 0 ? args[i + 1] : "controller.json";
        wmc.controllerJson = AssetDatabase.LoadAssetAtPath<TextAsset>("Assets/CA3/Data/" + json);
        // controller.json -> L4_..., controller_l41.json -> L41_..., controller_l42.json -> L42_...
        string tag = json == "controller.json" ? "L4" : Path.GetFileNameWithoutExtension(json).Replace("controller_", "").ToUpper();
        SessionState.SetString("CA3_L4_OUT", tag + "_unity_controller.csv");
        SessionState.SetBool(Flag, true);
        EditorApplication.EnterPlaymode();
    }

    static unsafe void Tick()
    {
        if (done || !EditorApplication.isPlaying || !MjScene.InstanceExists || MjScene.Instance.Data == null)
            return;
        if (!hooked)
        {
            MjScene.Instance.preUpdateEvent += Log;
            hooked = true;
        }
        if (n < Steps)
            return;
        done = true;
        Directory.CreateDirectory("TestResults");
        string outName = SessionState.GetString("CA3_L4_OUT", "L4_unity_controller.csv");
        File.WriteAllText("TestResults/" + outName, sb.ToString());
        var wmc = Object.FindAnyObjectByType<WorldModelController>();
        Debug.Log($"[L4] {wmc.controllerJson.name}: logged {n} steps to {outName}, enabled={wmc.enabled}, " +
                  $"actionSmoothing={wmc.actionSmoothing}, maxDeltaPerStep={wmc.maxDeltaPerStep}, actions={wmc.agent.ActionSize}");
        SessionState.EraseBool(Flag);
        EditorApplication.update -= Tick;
        EditorApplication.Exit(0);
    }

    static unsafe void Log(object sender, MjStepArgs e)
    {
        if (n >= Steps)
            return;
        var goals = Object.FindAnyObjectByType<ArmJointController>();
        var agent = goals.agent;
        var s = agent.Observe();
        var row = new float[24];
        System.Array.Copy(s, row, 12);
        System.Array.Copy(goals.CurrentTarget, 0, row, 12, 6);
        for (int i = 0; i < 6; i++)
            row[18 + i] = (float)e.data->ctrl[agent.actuators[i].MujocoId];
        for (int i = 0; i < row.Length; i++)
            sb.Append(i > 0 ? "," : "").Append(row[i].ToString("G9", CultureInfo.InvariantCulture));
        sb.Append('\n');
        n++;
    }
}
