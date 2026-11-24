// L5 batchmode check of the SAVED scene (nothing is added): IK alone, then the full loop
// TrackingTarget -> IkGoalProvider -> WorldModelController (controller_l42.json) -> ArmAgent -> MuJoCo.
// Unity.exe -batchmode -projectPath . -executeMethod L5Test.Run [-theta0] [-nodriver]
// The test only moves TrackingTarget (as the Scene View Move tool would) and logs.
// Outputs (TestResults/): L5_ik.txt, L5_loop[_theta0|_nodriver].csv / .txt, L5_wm_unity.csv, L5_shot_k.png
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Mujoco;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

[InitializeOnLoad]
public static class L5Test
{
    const string Flag = "CA3_L5_TEST";
    const int HoldSteps = 1000; // 2 s per target

    // joint poses -> reachable targets via FK (collision-free, joint5 inside the IK range; checked in Python)
    static readonly (string name, float[] q)[] IkPoses =
    {
        ("home", new[] { 0f, 0f, 1.57f, 0f, 1.57f, 0f }),
        ("near home", new[] { 0.15f, 0.1f, 1.7f, 0f, 1.45f, 0f }),
        ("left", new[] { 1.0f, 0.5f, 2.0f, 0f, 1.2f, 0f }),
        ("right", new[] { -1.0f, 0.5f, 2.0f, 0f, 1.2f, 0f }),
        ("high", new[] { 0.5f, -0.3f, 2.3f, 0f, 1.0f, 0f }),
        ("low", new[] { 0f, 0.7f, 0.9f, 0f, 1.2f, 0f }),
        ("far", new[] { 0.1f, 0.8f, 2.4f, 0f, 1.0f, 0f }),
        ("back", new[] { 2.5f, 0.2f, 1.8f, 0f, 1.3f, 0f }),
    };
    static readonly string[] LoopTargets = { "left", "right", "high", "low", "far", "near home" };
    static readonly Vector3 Unreachable = new(0.9f, 0.5f, 0.3f); // ~1 m from the base, Lite 6 reach is ~0.6 m

    static bool hooked, done, ikDone;
    static int n, k, phase = -1;
    static readonly StringBuilder csv = new(), wm = new(), txt = new();
    static readonly List<Vector3> targets = new();
    static readonly List<string> names = new();
    static string tag;
    static float wallStart;

    static L5Test()
    {
        if (SessionState.GetBool(Flag, false))
            EditorApplication.update += Tick;
    }

    public static void Run()
    {
        var args = System.Environment.GetCommandLineArgs();
        bool theta0 = args.Contains("-theta0"), noDriver = args.Contains("-nodriver");
        EditorSceneManager.OpenScene("Assets/Scenes/CA3.unity");
        // only the debugging switches of the saved components are changed, in memory
        Object.FindAnyObjectByType<WorldModelController>().thetaZero = theta0;
        if (noDriver)
            Object.FindAnyObjectByType<WorldModelDriver>().enabled = false;
        SessionState.SetString("CA3_L5_TAG", theta0 ? "_theta0" : noDriver ? "_nodriver" : "");
        SessionState.SetBool(Flag, true);
        EditorApplication.EnterPlaymode();
    }

    static string F(float v) => v.ToString("G7", CultureInfo.InvariantCulture);
    static string V(Vector3 v) => $"({v.x:F4}, {v.y:F4}, {v.z:F4})";
    static string Q(float[] q) => "[" + string.Join(" ", q.Select(x => x.ToString("+0.000;-0.000", CultureInfo.InvariantCulture))) + "]";

    static unsafe void Tick()
    {
        if (done || !EditorApplication.isPlaying || !MjScene.InstanceExists || MjScene.Instance.Data == null)
            return;
        var ik = Object.FindAnyObjectByType<IkGoalProvider>();
        if (!hooked)
        {
            tag = SessionState.GetString("CA3_L5_TAG", "");
            MjScene.Instance.preUpdateEvent += Log;
            hooked = true;
        }
        if (!ikDone && n > 10) // the IK has made its MjData by now
        {
            IkAlone(ik);
            ikDone = true;
        }
        if (phase < LoopTargets.Length + 1)
            return;
        done = true;
        Directory.CreateDirectory("TestResults");
        File.WriteAllText($"TestResults/L5_loop{tag}.csv", csv.ToString());
        File.WriteAllText($"TestResults/L5_loop{tag}.txt", txt.ToString());
        if (tag == "")
            File.WriteAllText("TestResults/L5_wm_unity.csv", wm.ToString());
        Debug.Log("[L5] done\n" + txt);
        SessionState.EraseBool(Flag);
        EditorApplication.update -= Tick;
        EditorApplication.Exit(0);
    }

    // A) IK only, on the IK's own MjData: FK of known poses -> target -> IK from home (and from the pose's neighbour).
    static void IkAlone(IkGoalProvider ik)
    {
        var home = IkPoses[0].q;
        var sb = new StringBuilder("L5 IK test (DLS, position only, IK's own MjData; target = FK(pose) in Unity world, m)\n");
        var agent = ik.agent;
        var real = agent.Observe().Take(6).ToArray();
        sb.AppendLine($"coordinate check at the current pose: IK FK {V(ik.ForwardKinematics(real))}, MuJoCo site_xpos " +
                      $"{V(ik.RealEndEffector())}, attachment_site Transform {V(agent.endEffector.position)}");
        var sol = new float[6];
        var errs = new List<float>();
        foreach (var (name, q) in IkPoses)
        {
            var t = ik.ForwardKinematics(q);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            float e = ik.Solve(t, home, sol, 200);
            sw.Stop();
            float check = Vector3.Distance(ik.ForwardKinematics(sol), t);
            errs.Add(check);
            sb.AppendLine($"  {name,-10} target {V(t)}  IK q {Q(sol)}  error {1000 * check:F2} mm  " +
                          $"({(e < ik.tolerance ? "converged" : "NOT converged")}, {sw.Elapsed.TotalMilliseconds:F2} ms)  pose q {Q(q)}");
            sb.AppendLine($"     ikq,{name}," + string.Join(",", sol.Select(F)));
        }
        {
            float e = ik.Solve(Unreachable, home, sol, 200);
            var reach = ik.ForwardKinematics(sol);
            bool finite = sol.All(x => !float.IsNaN(x) && !float.IsInfinity(x));
            sb.AppendLine($"  unreachable target {V(Unreachable)}: best IK q {Q(sol)}, reaches {V(reach)}, error {1000 * e:F1} mm, " +
                          $"converged={e < ik.tolerance}, finite={finite}");
            sb.AppendLine("     ikq,unreachable," + string.Join(",", sol.Select(F)));
        }
        sb.AppendLine($"reachable poses: max error {1000 * errs.Max():F3} mm, mean {1000 * errs.Average():F3} mm");
        foreach (var (name, q) in IkPoses)
            if (LoopTargets.Contains(name))
            {
                names.Add(name);
                targets.Add(ik.ForwardKinematics(q));
            }
        names.Add("unreachable");
        targets.Add(Unreachable);
        File.WriteAllText("TestResults/L5_ik.txt", sb.ToString());
        Debug.Log(sb.ToString());
    }

    // B) full loop: every HoldSteps physics steps the target moves; log every step.
    static unsafe void Log(object sender, MjStepArgs e)
    {
        var ik = Object.FindAnyObjectByType<IkGoalProvider>();
        var wmc = Object.FindAnyObjectByType<WorldModelController>();
        var driver = Object.FindAnyObjectByType<WorldModelDriver>();
        var agent = ik.agent;
        n++;
        if (!ikDone)
            return;
        if (phase < 0 || ++k >= HoldSteps)
        {
            k = 0;
            if (phase >= 0)
                Summary(ik, agent);
            phase++;
            if (phase < targets.Count)
                ik.target.position = targets[phase]; // the user moving TrackingTarget
            if (phase == 0)
                csv.Append("phase,step,target(3),ik_goal(6),ik_err,q(6),qd(6),ctrl(6),ee(3),ee_dist\n");
        }
        if (phase >= targets.Count)
            return;
        if (k == 150)
            Shot($"{phase}_moving"); // 0.3 s after the target moved
        var s = agent.Observe();
        var row = new List<float> { phase, n };
        var t = ik.target.position;
        row.AddRange(new[] { t.x, t.y, t.z });
        row.AddRange(ik.JointGoal);
        row.Add(ik.IkError);
        row.AddRange(s);
        for (int i = 0; i < 6; i++)
            row.Add((float)e.data->ctrl[agent.actuators[i].MujocoId]);
        var ee = ik.RealEndEffector();
        row.AddRange(new[] { ee.x, ee.y, ee.z, Vector3.Distance(ee, t) });
        csv.AppendLine(string.Join(",", row.Select(F)));

        // world model in Unity: one-step prediction from the real (state, ctrl used by this step), every 50 steps
        if (tag == "" && driver != null && driver.isActiveAndEnabled && n % 50 == 0)
        {
            var st = s.ToArray();
            var a = new float[6];
            for (int i = 0; i < 6; i++)
                a[i] = (float)e.data->ctrl[agent.actuators[i].MujocoId];
            var pred = new float[12];
            var sw = System.Diagnostics.Stopwatch.StartNew();
            driver.Predict(st, a, pred);
            sw.Stop();
            wm.AppendLine(string.Join(",", st.Concat(a).Concat(pred).Select(F)) + "," + F((float)sw.Elapsed.TotalMilliseconds));
        }
        if (phase == 0 && k == 0)
            wallStart = Time.realtimeSinceStartup;
        if (phase == 0 && k == 0)
            txt.Append($"L5 loop test{tag}: controller {wmc.controllerJson.name}, thetaZero={wmc.thetaZero}, " +
                       $"smoothing {wmc.actionSmoothing} / {wmc.maxDeltaPerStep}, goals from " +
                       $"{(wmc.Goal == ik.JointGoal ? "IkGoalProvider" : "ArmJointController")}, " +
                       $"driver enabled={driver != null && driver.isActiveAndEnabled}, {HoldSteps} steps (2 s) per target\n");
    }

    static void Summary(IkGoalProvider ik, ArmAgent agent)
    {
        var s = agent.Observe();
        var q = s.Take(6).ToArray();
        float track = Mathf.Sqrt(q.Select((v, i) => (v - ik.JointGoal[i]) * (v - ik.JointGoal[i])).Average());
        float qd = s.Skip(6).Select(Mathf.Abs).Max();
        txt.AppendLine($"  #{phase} {names[phase]}: target {V(ik.target.position)}");
        txt.AppendLine($"     IK goal {Q(ik.JointGoal)}  IK error {1000 * ik.IkError:F2} mm{(ik.Converged ? "" : " (not converged)")}");
        txt.AppendLine($"     real q  {Q(q)}  joint tracking RMSE {track:F4} rad, max |qd| {qd:F3} rad/s");
        txt.AppendLine($"     end effector {V(ik.RealEndEffector())}  distance to target {1000 * ik.EeError:F2} mm  " +
                       $"(attachment_site Transform {V(agent.endEffector.position)}, marker {V(ik.eeMarker.position)})");
        Shot(phase.ToString());
        if (phase == targets.Count - 1)
            txt.AppendLine($"real-time factor (simulated / wall time, batchmode with rendering): " +
                           $"{targets.Count * HoldSteps * 0.002f / (Time.realtimeSinceStartup - wallStart):F2}");
    }

    static void Shot(string k)
    {
        var cam = Camera.main;
        if (cam == null || SystemInfo.graphicsDeviceType == UnityEngine.Rendering.GraphicsDeviceType.Null)
            return;
        var rt = new RenderTexture(960, 540, 24);
        cam.targetTexture = rt;
        cam.aspect = 960f / 540f;
        cam.Render();
        RenderTexture.active = rt;
        var tex = new Texture2D(960, 540, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, 960, 540), 0, 0);
        cam.targetTexture = null;
        RenderTexture.active = null;
        File.WriteAllBytes($"TestResults/L5_shot{tag}_{k}.png", tex.EncodeToPNG());
        Object.DestroyImmediate(rt);
    }
}
