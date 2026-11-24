// L1 batchmode test for ArmAgent.Observe / SetAction.
// Unity.exe -batchmode -nographics -projectPath . -executeMethod L1ArmAgentTest.Run -logFile Logs/L1.log
// Results: TestResults/L1_ArmAgent.txt
using System.IO;
using System.Linq;
using System.Text;
using Mujoco;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

[InitializeOnLoad]
public static class L1ArmAgentTest
{
    const string ScenePath = "Assets/Scenes/CA3.unity";
    const string Flag = "CA3_L1_TEST";
    const string ResultPath = "TestResults/L1_ArmAgent.txt";
    static readonly float[] Home = { 0f, 0f, 1.57f, 0f, 1.57f, 0f };
    const float StepTarget = 0.5f; // new target for joint1 in phase 2

    static readonly StringBuilder report = new StringBuilder();
    static int stage;
    static bool pass = true, sampled1, sampled2;
    static float[] before;
    static float peakQ3, peakQ5, peakQ1, peakQd1, settleQd, othersDrift;

    static L1ArmAgentTest()
    {
        if (SessionState.GetBool(Flag, false))
            EditorApplication.update += Tick;
    }

    // One-time: put ArmAgent on the lite6 root and save the scene.
    public static void Setup()
    {
        var scene = EditorSceneManager.OpenScene(ScenePath);
        var root = GameObject.Find("lite6");
        var agent = root.GetComponent<ArmAgent>();
        if (agent == null)
            agent = root.AddComponent<ArmAgent>();
        agent.InitializeJointsAndActuators();
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
    }

    public static void Run()
    {
        EditorSceneManager.OpenScene(ScenePath);
        SessionState.SetBool(Flag, true);
        EditorApplication.EnterPlaymode();
    }

    static unsafe void Tick()
    {
        if (!EditorApplication.isPlaying || !MjScene.InstanceExists || MjScene.Instance.Data == null)
            return;
        var agent = Object.FindAnyObjectByType<ArmAgent>();
        var data = MjScene.Instance.Data;
        double t = data->time;
        var s = agent.Observe();
        bool finite = s.All(float.IsFinite);
        if (!finite) Check("state finite", false, $"t={t:F3} {Fmt(s)}");

        switch (stage)
        {
            case 0:
                Line($"L1 ArmAgent test  (fixedDeltaTime={Time.fixedDeltaTime}, mj timestep={MjScene.Instance.Model->opt.timestep})");
                Check("sizes", s.Length == 12 && agent.StateSize == 12 && agent.ActionSize == 6 && agent.JointCount == 6,
                    $"Observe().Length={s.Length} StateSize={agent.StateSize} ActionSize={agent.ActionSize} JointCount={agent.JointCount}");
                bool order = true;
                for (int i = 0; i < 6; i++)
                {
                    var j = agent.joints[i]; var a = agent.actuators[i];
                    order &= j.name == "joint" + (i + 1) && j.QposAddress == i && a.name == "act" + (i + 1) && a.Joint == j;
                    Line($"  [{i}] {j.name} qposadr={j.QposAddress} dofadr={j.DofAddress}  <-  {a.name} (Joint={a.Joint.name})");
                }
                Check("joint/actuator order", order, "action[i] -> act(i+1) -> joint(i+1) -> state[i], state[6+i]");
                Check("end effector", agent.endEffectorBody.name == "link6" && agent.endEffector.name == "attachment_site",
                    $"{agent.endEffectorBody.name}/{agent.endEffector.name}");
                Line($"t={t:F3} initial state {Fmt(s)}");
                agent.SetAction(Home);
                Line($"SetAction(Home={Fmt(Home)})");
                stage = 1;
                break;

            case 1: // approach Home for 3 s
                peakQ3 = Mathf.Max(peakQ3, s[2]);
                peakQ5 = Mathf.Max(peakQ5, s[4]);
                if (t >= 0.30 && !sampled1) { sampled1 = true; Line($"t={t:F3} moving     {Fmt(s)}"); }
                if (t > 2.5) settleQd = Mathf.Max(settleQd, MaxAbs(s, 6, 12));
                if (t >= 3.0)
                {
                    Line($"t={t:F3} at Home    {Fmt(s)}");
                    float err = Enumerable.Range(0, 6).Max(i => Mathf.Abs(s[i] - Home[i]));
                    Check("reaches Home", err < 0.01f, $"max|q-home|={err:F4} rad");
                    Check("settled (no oscillation)", settleQd < 1e-3f, $"max|qd| over t=[2.5,3.0]s = {settleQd:E2} rad/s");
                    Line($"  overshoot: peak q3={peakQ3:F4} (target 1.57), peak q5={peakQ5:F4} (target 1.57)");
                    Check("ctrl reached MuJoCo", Enumerable.Range(0, 6).All(i => Mathf.Abs((float)data->ctrl[agent.actuators[i].MujocoId] - Home[i]) < 1e-6f),
                        "data->ctrl == Home");
                    Check("Observe matches plugin qpos", Enumerable.Range(0, 6).All(i => Mathf.Abs(s[i] - (float)agent.joints[i].RawConfiguration) < 1e-6f),
                        "state[i] == joint.RawConfiguration");
                    before = (float[])s.Clone();
                    var target = (float[])Home.Clone();
                    target[0] = StepTarget;
                    agent.SetAction(target);
                    Line($"SetAction(joint1 -> {StepTarget}, others Home)");
                    stage = 2;
                }
                break;

            case 2: // only joint1 should move, for 3 s
                peakQ1 = Mathf.Max(peakQ1, s[0]);
                peakQd1 = Mathf.Max(peakQd1, Mathf.Abs(s[6]));
                for (int i = 1; i < 6; i++) othersDrift = Mathf.Max(othersDrift, Mathf.Abs(s[i] - before[i]));
                if (t >= 3.10 && !sampled2) { sampled2 = true; Line($"t={t:F3} joint1 moving {Fmt(s)}"); }
                if (t >= 6.0)
                {
                    Line($"t={t:F3} final      {Fmt(s)}");
                    Check("joint1 tracks new target", Mathf.Abs(s[0] - StepTarget) < 0.01f,
                        $"q1={s[0]:F4} (target {StepTarget}), peak q1={peakQ1:F4}, peak |qd1|={peakQd1:F3} rad/s");
                    Check("other joints unchanged", othersDrift < 0.01f, $"max drift of q2..q6 = {othersDrift:F4} rad");
                    Check("settled", MaxAbs(s, 6, 12) < 1e-3f, $"max|qd|={MaxAbs(s, 6, 12):E2} rad/s");
                    Line(pass ? "RESULT: PASS" : "RESULT: FAIL");
                    Directory.CreateDirectory(Path.GetDirectoryName(ResultPath));
                    File.WriteAllText(ResultPath, report.ToString());
                    SessionState.EraseBool(Flag);
                    EditorApplication.update -= Tick;
                    EditorApplication.Exit(pass ? 0 : 1);
                }
                break;
        }
    }

    static void Check(string name, bool ok, string detail)
    {
        pass &= ok;
        Line($"{(ok ? "PASS" : "FAIL")}  {name}: {detail}");
    }

    static void Line(string text)
    {
        report.AppendLine(text);
        Debug.Log("[L1] " + text);
    }

    static float MaxAbs(float[] v, int from, int to) =>
        Enumerable.Range(from, to - from).Max(i => Mathf.Abs(v[i]));

    static string Fmt(float[] v) =>
        "[" + string.Join(", ", v.Select(x => x.ToString("F4"))) + "]";
}
