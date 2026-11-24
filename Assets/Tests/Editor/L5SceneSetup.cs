// L5: builds the live-tracking setup in Assets/Scenes/CA3.unity and SAVES the scene.
// Unity.exe -batchmode -projectPath . -executeMethod L5SceneSetup.Run      (or menu CA3 > L5 Scene Setup)
// Safe to run again: existing objects / components are reused.
using Mujoco;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class L5SceneSetup
{
    const string Data = "Assets/CA3/Data/";
    const string Mats = "Assets/CA3/Materials/";

    [MenuItem("CA3/L5 Scene Setup")]
    public static void Run()
    {
        var scene = EditorSceneManager.OpenScene("Assets/Scenes/CA3.unity");
        var root = GameObject.Find("lite6");
        var agent = root.GetComponent<ArmAgent>();
        var goals = root.GetComponent<ArmJointController>();
        goals.emitActions = false; // goals come from the IK (WorldModelController also does this at Start)

        if (!AssetDatabase.IsValidFolder("Assets/CA3/Materials"))
            AssetDatabase.CreateFolder("Assets/CA3", "Materials");
        var red = Material("Target", "Standard", new Color(0.9f, 0.15f, 0.1f));
        var green = Material("EndEffector", "Standard", new Color(0.1f, 0.8f, 0.2f));
        var ghostMat = Material("Ghost", "Legacy Shaders/Transparent/Diffuse", new Color(0.3f, 0.7f, 1f, 0.3f));
        var lineMat = Material("Line", "Sprites/Default", Color.white);

        var target = Marker("TrackingTarget", 0.05f, red, new Vector3(0.25f, 0.42f, 0f));
        var ee = Marker("EndEffectorMarker", 0.02f, green, Vector3.zero);
        var ghost = Ghost(root, ghostMat);
        var forecast = Line("ArmForecast", 0.006f, new Color(0.2f, 0.8f, 1f), lineMat);
        var deviation = Line("ArmDeviation", 0.004f, Color.yellow, lineMat);

        var ik = Get<IkGoalProvider>(root);
        var wmc = Get<WorldModelController>(root);
        var driver = Get<WorldModelDriver>(root);

        ik.agent = agent;
        ik.joints = goals;
        ik.policy = wmc;
        ik.target = target;
        ik.eeMarker = ee;

        wmc.agent = agent;
        wmc.goals = goals;
        wmc.ik = ik;
        wmc.controllerJson = AssetDatabase.LoadAssetAtPath<TextAsset>(Data + "controller_l42.json");
        wmc.actionSmoothing = 0.2f;
        wmc.maxDeltaPerStep = 0.05f;
        wmc.thetaZero = false;

        driver.agent = agent;
        driver.controller = goals;
        driver.policy = wmc;
        driver.modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(Data + "dynamics_multistep.onnx");
        driver.normJson = AssetDatabase.LoadAssetAtPath<TextAsset>(Data + "norm_multistep.json");
        driver.ghostArm = ghost;
        driver.forecast = forecast;
        driver.deviation = deviation;
        driver.ghostMaterial = ghostMat;
        driver.lineMaterial = lineMat;
        // one Predict is ~0.1 ms in Unity: forecast 200 steps (0.4 s, the controller's training horizon) every 50 steps
        driver.horizon = 200;
        driver.forecastInterval = 50;

        // camera: front-right view of the arm's workspace
        var cam = Camera.main.transform;
        cam.position = new Vector3(0.8f, 0.65f, -0.6f);
        cam.LookAt(new Vector3(0.12f, 0.38f, 0f));
        EditorUtility.SetDirty(cam);

        foreach (var c in new Object[] { goals, ik, wmc, driver })
            EditorUtility.SetDirty(c);
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        AssetDatabase.SaveAssets();
        Debug.Log($"[L5] scene saved: {scene.path}; controller {wmc.controllerJson.name}, model {driver.modelAsset.name}, " +
                  $"norm {driver.normJson.name}, ghost parts under {ghost.name}");
        if (Application.isBatchMode)
            EditorApplication.Exit(0);
    }

    static T Get<T>(GameObject go) where T : Component
    {
        var c = go.GetComponent<T>();
        return c != null ? c : go.AddComponent<T>();
    }

    static Material Material(string name, string shader, Color color)
    {
        string path = Mats + name + ".mat";
        var m = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (m == null)
        {
            m = new Material(Shader.Find(shader));
            AssetDatabase.CreateAsset(m, path);
        }
        m.color = color;
        EditorUtility.SetDirty(m);
        return m;
    }

    static Transform Marker(string name, float size, Material mat, Vector3 position)
    {
        var go = GameObject.Find(name);
        if (go == null)
        {
            go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            go.name = name;
            Object.DestroyImmediate(go.GetComponent<Collider>()); // visual only
            go.transform.position = position;
        }
        go.transform.localScale = Vector3.one * size;
        go.GetComponent<MeshRenderer>().sharedMaterial = mat;
        return go.transform;
    }

    static LineRenderer Line(string name, float width, Color color, Material mat)
    {
        var go = GameObject.Find(name);
        if (go == null)
            go = new GameObject(name);
        var line = Get<LineRenderer>(go);
        line.widthMultiplier = width;
        line.startColor = line.endColor = color;
        line.sharedMaterial = mat;
        line.useWorldSpace = true;
        line.positionCount = 2;
        return line;
    }

    // Visual-only copy of the arm for WorldModelDriver (no MuJoCo components, no scripts, no colliders).
    static Transform Ghost(GameObject root, Material mat)
    {
        var old = GameObject.Find(root.name + " (Ghost)");
        if (old != null)
            Object.DestroyImmediate(old);
        var ghost = Object.Instantiate(root);
        ghost.name = root.name + " (Ghost)";
        for (int pass = 0; pass < 5; pass++) // some components require others; remove until none are left
            foreach (var b in ghost.GetComponentsInChildren<MonoBehaviour>(true))
                Object.DestroyImmediate(b);
        foreach (var c in ghost.GetComponentsInChildren<Collider>(true))
            Object.DestroyImmediate(c);
        foreach (var r in ghost.GetComponentsInChildren<MeshRenderer>(true))
        {
            r.sharedMaterial = mat;
            r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        }
        return ghost.transform;
    }
}
