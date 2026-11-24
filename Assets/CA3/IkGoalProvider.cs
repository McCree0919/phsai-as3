using Mujoco;
using UnityEngine;

// L5: position-only Cartesian IK for attachment_site (damped least squares).
// TrackingTarget (Unity world) -> 6 joint goals (rad) -> WorldModelController (learned policy) -> ArmAgent -> MuJoCo.
// The IK runs on its own MjData (mj_makeData), so the simulated arm's qpos is never touched.
[DefaultExecutionOrder(-8)] // after ArmJointController (-10), before WorldModelController (-5)
public class IkGoalProvider : MonoBehaviour
{
    public ArmAgent agent;
    public ArmJointController joints; // home targets + joint ranges (ROBOT_CONFIG.md)
    public WorldModelController policy; // only for the on-screen text and the controller toggle
    public Transform target;
    public Transform eeMarker; // optional: follows the real attachment_site

    public float damping = 0.05f;    // DLS lambda (m)
    public int iterations = 30;      // max DLS iterations per FixedUpdate (warm-started from the last solution)
    public float tolerance = 0.0005f; // m (0.5 mm)
    public float maxStep = 0.2f;     // rad per iteration
    public float nullGain = 0.05f;   // pull toward home in the null space (3 position equations, 6 joints)
    // link4 / link6 collision meshes touch for |joint5| > ~1.63 rad (MuJoCo contact test, ROBOT_CONFIG.md).
    // The IK keeps joint5 inside this range; the MuJoCo collision itself is unchanged.
    public float joint5Limit = 1.6f;
    public float moveSpeed = 0.25f;  // m/s, keyboard

    float[] goal, q, alt, lo, hi;
    double[] jacp;
    unsafe MujocoLib.mjData_* ik;
    int siteId;

    public float[] JointGoal => goal;
    public bool Converged { get; private set; }
    public float IkError { get; private set; } // |FK(goal) - target| (m): IK solution error
    public float EeError { get; private set; } // |real attachment_site - target| (m): what the arm actually reached

    void Awake()
    {
        if (agent == null)
            agent = GetComponent<ArmAgent>();
        if (joints == null)
            joints = GetComponent<ArmJointController>();
        int n = 6;
        goal = (float[])joints.homeTargets.Clone(); // the controller tracks home until the first solve
        q = new float[n];
        alt = new float[n];
        lo = new float[n];
        hi = new float[n];
        for (int i = 0; i < n; i++)
        {
            lo[i] = joints.jointRanges[i].x;
            hi[i] = joints.jointRanges[i].y;
        }
        lo[4] = Mathf.Max(lo[4], -joint5Limit);
        hi[4] = Mathf.Min(hi[4], joint5Limit);
    }

    unsafe void FixedUpdate()
    {
        if (target == null || !MjScene.InstanceExists || MjScene.Instance.Data == null)
            return;
        if (ik == null)
        {
            ik = MujocoLib.mj_makeData(MjScene.Instance.Model);
            siteId = agent.endEffector.GetComponent<MjSite>().MujocoId;
            jacp = new double[3 * MjScene.Instance.Model->nv];
        }
        // warm start from the last solution (smooth goals while the target moves); if that gets stuck
        // (joint limit / local minimum), also try from home and keep the better one
        IkError = Solve(target.position, goal, goal, iterations);
        if (IkError >= tolerance)
        {
            float e = Solve(target.position, joints.homeTargets, alt, iterations);
            if (e < IkError - 0.001f)
            {
                IkError = e;
                System.Array.Copy(alt, goal, 6);
            }
        }
        Converged = IkError < tolerance;
        EeError = Vector3.Distance(RealEndEffector(), target.position);
    }

    void Update()
    {
        if (target != null)
        {
            // Game View: WASD = horizontal, Q / E = down / up (Scene View: move TrackingTarget with the Move tool).
            var v = new Vector3(Axis(KeyCode.D, KeyCode.A), Axis(KeyCode.E, KeyCode.Q), Axis(KeyCode.W, KeyCode.S));
            target.position += moveSpeed * Time.deltaTime * v;
        }
        if (policy != null && Input.GetKeyDown(KeyCode.T))
            policy.thetaZero = !policy.thetaZero;
        if (eeMarker != null && agent.endEffector != null)
            eeMarker.position = agent.endEffector.position;
    }

    static float Axis(KeyCode plus, KeyCode minus)
    {
        return (Input.GetKey(plus) ? 1f : 0f) - (Input.GetKey(minus) ? 1f : 0f);
    }

    // DLS: dq = J^T (J J^T + lambda^2 I)^-1 e  +  nullGain (I - J^T (J J^T)^-1 J)(home - q), clamped to the joint ranges.
    // (the null-space projection uses a tiny regularisation, so the posture pull does not move the end effector)
    // Starts from seed, writes the best joints to result, returns the remaining position error (m).
    public unsafe float Solve(Vector3 targetWorld, float[] seed, float[] result, int maxIterations)
    {
        var m = MjScene.Instance.Model;
        System.Array.Copy(seed, q, 6);
        Vector3 t = MjEngineTool.MjVector3(targetWorld); // Unity (x, y, z) -> MuJoCo (x, z, y)
        float err = 0f;
        for (int it = 0; ; it++)
        {
            Vector3 e = t - SitePosition(q);
            err = e.magnitude;
            if (err < tolerance || it >= maxIterations)
                break;
            MujocoLib.mj_comPos(m, ik);
            fixed (double* jp = jacp)
                MujocoLib.mj_jacSite(m, ik, jp, null, siteId);
            Vector3 y = JJt(damping * damping).inverse.MultiplyVector(e); // y = (J J^T + lambda^2 I)^-1 e
            // null space: z = home - q, remove the part J^+ J z that would move the end effector
            var Jz = Vector3.zero;
            for (int j = 0; j < 6; j++)
                for (int r = 0; r < 3; r++)
                    Jz[r] += J(r, j) * (joints.homeTargets[j] - q[j]);
            Vector3 w = JJt(1e-4f).inverse.MultiplyVector(Jz);
            for (int j = 0; j < 6; j++)
            {
                float dq = 0f, back = 0f;
                for (int r = 0; r < 3; r++)
                {
                    dq += J(r, j) * y[r];
                    back += J(r, j) * w[r];
                }
                dq += nullGain * (joints.homeTargets[j] - q[j] - back);
                q[j] = Mathf.Clamp(q[j] + Mathf.Clamp(dq, -maxStep, maxStep), lo[j], hi[j]);
            }
        }
        System.Array.Copy(q, result, 6);
        return err;
    }

    // J J^T + reg I (3x3, in a Matrix4x4 with [3,3] = 1)
    Matrix4x4 JJt(float reg)
    {
        var A = new Matrix4x4();
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
            {
                float s = r == c ? reg : 0f;
                for (int j = 0; j < 6; j++)
                    s += J(r, j) * J(c, j);
                A[r, c] = s;
            }
        A[3, 3] = 1f;
        return A;
    }

    float J(int row, int joint)
    {
        return (float)jacp[row * jacp.Length / 3 + agent.joints[joint].DofAddress];
    }

    // FK on the IK's own MjData (MuJoCo world coordinates).
    unsafe Vector3 SitePosition(float[] joints6)
    {
        for (int i = 0; i < 6; i++)
            ik->qpos[agent.joints[i].QposAddress] = joints6[i];
        MujocoLib.mj_kinematics(MjScene.Instance.Model, ik);
        double* p = ik->site_xpos + 3 * siteId;
        return new Vector3((float)p[0], (float)p[1], (float)p[2]);
    }

    // FK in Unity world coordinates (used by the L5 test to make reachable targets).
    public Vector3 ForwardKinematics(float[] joints6)
    {
        return MjEngineTool.UnityVector3(SitePosition(joints6));
    }

    public unsafe Vector3 RealEndEffector()
    {
        return MjEngineTool.UnityVector3(MjEngineTool.MjVector3AtEntry(MjScene.Instance.Data->site_xpos, siteId));
    }

    unsafe void OnGUI()
    {
        if (target == null)
            return;
        string mode = policy == null ? "-" : policy.thetaZero ? "theta = 0 (raw = IK goal)" : "L4.2 learned (controller_l42.json)";
        GUI.Label(new Rect(10, 10, 700, 110),
            $"target {target.position:F3}   IK error {1000f * IkError:F1} mm {(Converged ? "" : "(not converged)")}\n" +
            $"end effector {(MjScene.InstanceExists && ik != null ? RealEndEffector().ToString("F3") : "-")}   " +
            $"distance to target {1000f * EeError:F1} mm\n" +
            $"controller: {mode}\n" +
            "WASD / Q E: move target    T: toggle controller    (or move TrackingTarget in the Scene View)");
    }

    unsafe void OnDestroy()
    {
        if (ik != null)
            MujocoLib.mj_deleteData(ik);
        ik = null;
    }
}
