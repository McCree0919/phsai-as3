using System.Collections.Generic;
using Mujoco;
using Unity.InferenceEngine;
using UnityEngine;

[DefaultExecutionOrder(5)] // L5: OnEnable after ArmAgent.Awake (agent.Ready), FixedUpdate after WorldModelController
public class WorldModelDriver : MonoBehaviour
{
    public ArmAgent agent;
    public ArmJointController controller;
    public WorldModelController policy;
    public ModelAsset modelAsset;
    public TextAsset normJson;
    public Transform ghostArm;
    public Transform ghostMarker;
    public LineRenderer forecast;
    public LineRenderer deviation;
    public Material ghostMaterial;
    public Material lineMaterial;

    public int horizon = 100;
    public int ghostInterval = 3000;
    [Range(0f, 0.25f)]
    public float softCorrection = 0.05f;
    public float maxJointError = 0.2f;
    public int forecastInterval = 10;
    public bool drawForecast = true;

    int FeatureSize => agent != null ? agent.StateSize + agent.ActionSize : 0;

    struct GhostPart
    {
        public MjGeom source;
        public Transform ghost;
        public Matrix4x4 geomToMesh;
    }

    Worker worker;
    DynamicsNorm norm;
    Tensor<float> input;
    float[] features;
    float[] actionBuffer;
    float[] realState;
    double[] savedQ;
    GhostPart[] ghostParts = System.Array.Empty<GhostPart>();
    float[] ghostState;
    float[] simulated;
    float[] predicted;
    float[] forecastScratch;
    Vector3[] points;
    // Ghost copy of the control path, same as train_controller.controller_step (Unity 2-step delay):
    // ghostPrev = last smoothed action, ghostQueue = the 2 actions still waiting to reach data->ctrl.
    float[] ghostPrev;
    float[][] ghostQueue;
    float[] realPrev1, realPrev2; // the real arm's actions from 1 and 2 frames ago (for reanchoring)
    bool anchored;
    int ghostSteps;
    int forecastSteps;
    bool ready;
    bool ghostCalibrated;
    bool ownsGhostInstance;
    bool ownsForecast;
    bool ownsDeviation;
    Transform ghostPrefab;

    void Awake()
    {
        if (agent == null)
            agent = GetComponent<ArmAgent>();
        if (controller == null)
            controller = GetComponent<ArmJointController>();
        if (policy == null)
            policy = GetComponent<WorldModelController>();
        ghostPrefab = ghostArm;
    }

    void OnEnable()
    {
        if (ready)
            return;

        if (agent == null || modelAsset == null || normJson == null)
        {
            Debug.LogError("WorldModelDriver needs the agent, dynamics.onnx, and norm.json.");
            enabled = false;
            return;
        }

        if (!agent.Ready)
        {
            Debug.LogError("WorldModelDriver needs the agent to be ready.");
            enabled = false;
            return;
        }

        features = new float[FeatureSize];
        actionBuffer = new float[agent.ActionSize];
        realState = new float[agent.StateSize];
        savedQ = new double[agent.JointCount];
        ghostState = new float[agent.StateSize];
        simulated = new float[agent.StateSize];
        predicted = new float[agent.StateSize];
        forecastScratch = new float[agent.StateSize];
        ghostPrev = new float[agent.ActionSize];
        ghostQueue = new[] { new float[agent.ActionSize], new float[agent.ActionSize] };
        realPrev1 = new float[agent.ActionSize];
        realPrev2 = new float[agent.ActionSize];

        norm = JsonUtility.FromJson<DynamicsNorm>(normJson.text);
        if (norm == null ||
            norm.x_mean == null || norm.x_mean.Length != FeatureSize ||
            norm.y_mean == null || norm.y_mean.Length != agent.StateSize)
        {
            Debug.LogError(
                "norm.json does not match StateSize=" + agent.StateSize +
                " ActionSize=" + agent.ActionSize + ".");
            enabled = false;
            return;
        }

        worker = new Worker(ModelLoader.Load(modelAsset), BackendType.CPU);
        input = new Tensor<float>(new TensorShape(1, FeatureSize), features);
        CreateVisuals();
        ready = true;
    }

    unsafe void FixedUpdate()
    {
        if (!ready || !agent.Ready || !MjScene.InstanceExists || MjScene.Instance.Data == null)
            return;

        agent.CopyState(realState);
        if (!anchored)
        {
            // first frame: the arm is at rest and the ctrl holds the current pose
            System.Array.Copy(realState, realPrev1, agent.ActionSize);
            System.Array.Copy(realState, realPrev2, agent.ActionSize);
        }
        if (!anchored || ghostSteps >= ghostInterval || JointError(ghostState, realState) > maxJointError)
            Reanchor();

        FillAction(ghostState, actionBuffer);
        Predict(ghostState, actionBuffer, predicted);
        for (int i = 0; i < agent.StateSize; i++)
            ghostState[i] = predicted[i];
        float keep = 1f - softCorrection;
        for (int i = 0; i < agent.StateSize; i++)
            ghostState[i] = keep * ghostState[i] + softCorrection * realState[i];

        ghostSteps++;

        PlaceGhost(ghostState);
        if (drawForecast && forecastSteps % Mathf.Max(1, forecastInterval) == 0)
            ShowForecast(ghostState);
        forecastSteps++;
        ShowDeviation();
        // Remember the real arm's action history (this frame's action was set by WorldModelController, order -5)
        System.Array.Copy(realPrev1, realPrev2, agent.ActionSize);
        System.Array.Copy(agent.CurrentAction, realPrev1, agent.ActionSize);
    }

    void CreateVisuals()
    {
        horizon = Mathf.Max(1, horizon);
        forecastInterval = Mathf.Max(1, forecastInterval);
        points = new Vector3[horizon + 1];
        BuildGhostArm();

        if (forecast == null)
        {
            var go = new GameObject("ArmForecast");
            forecast = go.AddComponent<LineRenderer>();
            forecast.widthMultiplier = 0.01f;
            forecast.positionCount = points.Length;
            ownsForecast = true;
        }

        if (lineMaterial != null)
            forecast.sharedMaterial = lineMaterial;
        forecast.enabled = drawForecast;
        forecast.positionCount = points.Length;

        if (deviation == null)
        {
            var go = new GameObject("ArmDeviation");
            deviation = go.AddComponent<LineRenderer>();
            deviation.widthMultiplier = 0.008f;
            deviation.positionCount = 2;
            ownsDeviation = true;
        }
        if (lineMaterial != null)
            deviation.sharedMaterial = lineMaterial;
    }

    void BuildGhostArm()
    {
        if (ghostPrefab == null)
            ghostPrefab = ghostArm;

        if (ghostPrefab == null)
        {
            Debug.LogError(
                "WorldModelDriver needs a visual-only ghost arm prefab assigned to Ghost Arm.");
            ghostParts = System.Array.Empty<GhostPart>();
            return;
        }

        if (!ghostPrefab.gameObject.scene.IsValid())
        {
            ghostArm = InstantiateGhostVisual(ghostPrefab.gameObject);
            ownsGhostInstance = true;
        }
        else
        {
            ghostArm = ghostPrefab;
            StripGhostMuJoCo(ghostArm.gameObject);
            ownsGhostInstance = false;
        }

        if (ghostMarker == null)
        {
            var marker = new GameObject("ArmGhostEE");
            marker.transform.SetParent(ghostArm, false);
            ghostMarker = marker.transform;
        }

        if (ghostMaterial != null)
        {
            foreach (var renderer in ghostArm.GetComponentsInChildren<MeshRenderer>(true))
            {
                renderer.sharedMaterial = ghostMaterial;
                renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                renderer.receiveShadows = false;
            }
        }

        // Map real MjGeoms to the same-named transforms on the prefab
        var parts = new List<GhostPart>();
        foreach (var geom in GetComponentsInChildren<MjGeom>(true))
        {
            var renderer = geom.GetComponent<MeshRenderer>();
            var filter = geom.GetComponent<MeshFilter>();
            if (renderer == null || filter == null || filter.sharedMesh == null || !renderer.enabled)
                continue;

            var ghostTransform = FindChild(ghostArm, geom.name);
            if (ghostTransform == null)
                continue;

            parts.Add(new GhostPart
            {
                source = geom,
                ghost = ghostTransform,
                geomToMesh = Matrix4x4.identity,
            });
        }

        ghostParts = parts.ToArray();
        ghostCalibrated = false;
        if (ghostParts.Length == 0)
            Debug.LogWarning(
                "WorldModelDriver found no name-matched geoms between the real arm and ghost prefab.");
    }

    static Transform InstantiateGhostVisual(GameObject source)
    {
        bool restoreActive = source.activeSelf;
        if (restoreActive)
            source.SetActive(false);

        var instance = Instantiate(source);
        instance.name = source.name + " (Ghost)";

        if (restoreActive)
            source.SetActive(true);

        StripGhostMuJoCo(instance);
        instance.SetActive(true);
        return instance.transform;
    }

    static void StripGhostMuJoCo(GameObject root)
    {
        foreach (var meshFilter in root.GetComponentsInChildren<MjMeshFilter>(true))
            DestroyImmediate(meshFilter);

        foreach (var settings in root.GetComponentsInChildren<MjGlobalSettings>(true))
            DestroyImmediate(settings);

        foreach (var component in root.GetComponentsInChildren<MjComponent>(true))
            DestroyImmediate(component);

        foreach (var collider in root.GetComponentsInChildren<Collider>(true))
            collider.enabled = false;
    }

    static Transform FindChild(Transform root, string name)
    {
        if (root.name == name)
            return root;
        for (int i = 0; i < root.childCount; i++)
        {
            var found = FindChild(root.GetChild(i), name);
            if (found != null)
                return found;
        }
        return null;
    }

    // The action the MuJoCo step uses from this (ghost) state: the controller's new smoothed action enters the
    // 2-step queue and the action from 2 frames earlier comes out (ctrl[t+2] = smooth(ctrl[t+1], policy(s_t))).
    void FillAction(float[] state, float[] destination)
    {
        if (policy != null && policy.isActiveAndEnabled)
            policy.ActionFrom(state, policy.Goal, ghostPrev); // ghostPrev: previous action in, new action out
        else
            controller.ActionFrom(ghostPrev); // ArmJointController: raw joint goal
        var oldest = ghostQueue[0];
        System.Array.Copy(oldest, destination, agent.ActionSize);
        System.Array.Copy(ghostPrev, oldest, agent.ActionSize);
        ghostQueue[0] = ghostQueue[1];
        ghostQueue[1] = oldest;
    }

    // Snap the ghost back to the real arm, with the real control history.
    void Reanchor()
    {
        System.Array.Copy(realState, ghostState, agent.StateSize);
        System.Array.Copy(realPrev1, ghostPrev, agent.ActionSize);
        System.Array.Copy(realPrev2, ghostQueue[0], agent.ActionSize);
        System.Array.Copy(realPrev1, ghostQueue[1], agent.ActionSize);
        ghostSteps = 0;
        anchored = true;
    }

    float JointError(float[] ghost, float[] real)
    {
        float sum = 0f;
        for (int i = 0; i < agent.ActionSize; i++)
        {
            float error = ghost[i] - real[i];
            sum += error * error;
        }
        return Mathf.Sqrt(sum / agent.ActionSize);
    }

    // One world-model step (L3 contract, the ONNX graph has no normalization):
    // x = ([state, action] - x_mean) / x_std;  next = state + delta * y_std + y_mean.
    public void Predict(float[] state, float[] action, float[] next)
    {
        int s = agent.StateSize;
        for (int i = 0; i < FeatureSize; i++)
            features[i] = ((i < s ? state[i] : action[i - s]) - norm.x_mean[i]) / norm.x_std[i];
        input.Upload(features);
        worker.Schedule(input);
        var delta = (worker.PeekOutput() as Tensor<float>).DownloadToArray();
        for (int i = 0; i < s; i++)
            next[i] = state[i] + delta[i] * norm.y_std[i] + norm.y_mean[i];
    }

    unsafe void PlaceGhost(float[] state)
    {
        if (ghostParts.Length == 0)
            return;

        var scene = MjScene.Instance;
        var data = scene.Data;

        if (!ghostCalibrated)
            CalibrateGhostOffsets(data);

        // Restore the live joint positions after computing the ghost pose.
        for (int i = 0; i < agent.JointCount; i++)
            savedQ[i] = data->qpos[agent.joints[i].QposAddress];

        ApplyJointStateNoRestore(state, scene, data);

        for (int i = 0; i < ghostParts.Length; i++)
        {
            var part = ghostParts[i];
            var geomPose = GeomWorldMatrix(data, part.source.MujocoId);
            var meshPose = geomPose * part.geomToMesh;
            part.ghost.SetPositionAndRotation(meshPose.GetPosition(), meshPose.rotation);
        }

        if (ghostMarker != null && agent.endEffectorBody != null)
        {
            ghostMarker.position = MjEngineTool.UnityVector3(
                data->xpos + 3 * agent.endEffectorBody.MujocoId);
            ghostMarker.rotation = MjEngineTool.UnityQuaternion(
                MjEngineTool.MjQuaternionAtEntry(data->xquat, agent.endEffectorBody.MujocoId));
        }

        for (int i = 0; i < agent.JointCount; i++)
            data->qpos[agent.joints[i].QposAddress] = savedQ[i];
        MujocoLib.mj_kinematics(scene.Model, data);
    }

    unsafe void CalibrateGhostOffsets(MujocoLib.mjData_* data)
    {
        for (int i = 0; i < ghostParts.Length; i++)
        {
            var part = ghostParts[i];
            var geomPose = GeomWorldMatrix(data, part.source.MujocoId);
            var meshPose = part.source.transform.localToWorldMatrix;
            // Cache the mesh offset from the MuJoCo geometry frame.
            part.geomToMesh = geomPose.inverse * meshPose;
            ghostParts[i] = part;
        }

        ghostCalibrated = true;
    }

    unsafe static Matrix4x4 GeomWorldMatrix(MujocoLib.mjData_* data, int geomId)
    {
        var position = MjEngineTool.UnityVector3(MjEngineTool.MjVector3AtEntry(data->geom_xpos, geomId));
        var rotation = MjEngineTool.UnityQuaternionFromMatrix(
            MjEngineTool.MjMatrixAtEntry(data->geom_xmat, geomId));
        return Matrix4x4.TRS(position, rotation, Vector3.one);
    }

    void ShowDeviation()
    {
        if (deviation == null || ghostMarker == null || agent.endEffector == null)
            return;
        deviation.SetPosition(0, ghostMarker.position);
        deviation.SetPosition(1, agent.endEffector.position);
    }

    // Imagined end-effector path for the next `horizon` steps: world model + the same controller from the ghost state.
    unsafe void ShowForecast(float[] state)
    {
        var scene = MjScene.Instance;
        var data = scene.Data;
        // keep the ghost's own control state; the forecast runs on copies
        var savedPrev = (float[])ghostPrev.Clone();
        var savedQueue = new[] { (float[])ghostQueue[0].Clone(), (float[])ghostQueue[1].Clone() };
        for (int i = 0; i < agent.JointCount; i++)
            savedQ[i] = data->qpos[agent.joints[i].QposAddress];

        System.Array.Copy(state, forecastScratch, agent.StateSize);
        points[0] = EndEffectorFromJointsNoRestore(forecastScratch, scene, data);
        for (int k = 1; k < points.Length; k++)
        {
            FillAction(forecastScratch, actionBuffer);
            Predict(forecastScratch, actionBuffer, simulated);
            System.Array.Copy(simulated, forecastScratch, agent.StateSize);
            points[k] = EndEffectorFromJointsNoRestore(forecastScratch, scene, data);
        }
        forecast.positionCount = points.Length;
        forecast.SetPositions(points);

        // restore the real arm's kinematics and the ghost's control state
        for (int i = 0; i < agent.JointCount; i++)
            data->qpos[agent.joints[i].QposAddress] = savedQ[i];
        MujocoLib.mj_kinematics(scene.Model, data);
        System.Array.Copy(savedPrev, ghostPrev, agent.ActionSize);
        System.Array.Copy(savedQueue[0], ghostQueue[0], agent.ActionSize);
        System.Array.Copy(savedQueue[1], ghostQueue[1], agent.ActionSize);
    }

    unsafe void ApplyJointStateNoRestore(float[] state, MjScene scene, MujocoLib.mjData_* data)
    {
        for (int i = 0; i < agent.JointCount; i++)
            data->qpos[agent.joints[i].QposAddress] = state[i];
        MujocoLib.mj_kinematics(scene.Model, data);
    }

    unsafe Vector3 EndEffectorFromJointsNoRestore(float[] state, MjScene scene, MujocoLib.mjData_* data)
    {
        ApplyJointStateNoRestore(state, scene, data);
        return MjEngineTool.UnityVector3(
            data->xpos + 3 * agent.endEffectorBody.MujocoId);
    }

    void OnDisable()
    {
        DisposeModel();
        TeardownVisuals();
        ready = false;
    }

    void OnDestroy()
    {
        DisposeModel();
        TeardownVisuals();
        ready = false;
    }

    void DisposeModel()
    {
        if (input != null)
        {
            input.Dispose();
            input = null;
        }
        if (worker != null)
        {
            worker.Dispose();
            worker = null;
        }
    }

    void TeardownVisuals()
    {
        if (ownsForecast && forecast != null)
        {
            Destroy(forecast.gameObject);
            forecast = null;
            ownsForecast = false;
        }
        else if (forecast != null)
        {
            forecast.enabled = false;
        }

        if (ownsDeviation && deviation != null)
        {
            Destroy(deviation.gameObject);
            deviation = null;
            ownsDeviation = false;
        }
        else if (deviation != null)
        {
            deviation.enabled = false;
        }

        if (ownsGhostInstance && ghostArm != null)
        {
            Destroy(ghostArm.gameObject);
            ghostArm = ghostPrefab;
            ghostMarker = null;
            ownsGhostInstance = false;
        }
        else if (ghostArm != null)
        {
            ghostArm.gameObject.SetActive(false);
        }

        ghostParts = System.Array.Empty<GhostPart>();
        ghostCalibrated = false;
    }
}

[System.Serializable]
public class DynamicsNorm
{
    public float[] x_mean;
    public float[] x_std;
    public float[] y_mean;
    public float[] y_std;
}
