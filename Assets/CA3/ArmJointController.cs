using UnityEngine;


[DefaultExecutionOrder(-10)]
public class ArmJointController : MonoBehaviour
{
    public ArmAgent agent;
    public float holdSeconds = 3f;

    public Vector2[] jointRanges;

    public float[] homeTargets;

    [Range(0f, 1f)]
    public float randomScale = 0.35f;

    public bool emitActions = true;

    public int seed = 0;

    // L3.1 data mode (off = L2 step targets). On: action = smooth_action(prev, raw) as in train_controller.py.
    // Random targets alternate between A: raw = target, and B: raw = target + W [q - target, qd] + b
    // with a new random W ~ N(0, explorationStd), b ~ N(0, explorationStd / 2) per target (CMA-ES style).
    public bool smoothActions = false;
    public float explorationStd = 0.1f;
    const float ActionSmooth = 0.2f; // train_controller.ACTION_SMOOTH
    const float MaxDelta = 0.05f;    // train_controller.MAX_DELTA

    float[] target;
    float hold;
    bool ready;
    float[] smoothed;
    float[] W;
    float[] b;
    int targetCount;

    public float[] CurrentTarget => target;

    void Start()
    {
        if (agent == null)
            agent = GetComponent<ArmAgent>();
        Random.InitState(seed);
        target = new float[agent.ActionSize];
        PlaceTarget(true);
        ready = true;
    }

    void FixedUpdate()
    {
        if (!ready)
            return;

        hold -= Time.fixedDeltaTime;
        if (hold <= 0f)
            PlaceTarget(false);

        if (emitActions)
            agent.SetAction(smoothActions ? SmoothAction() : target);
    }

    float[] SmoothAction()
    {
        var s = agent.Observe();
        int n = target.Length;
        if (smoothed == null)
        {
            smoothed = new float[n];
            System.Array.Copy(s, smoothed, n); // start from the current joint positions
        }
        for (int i = 0; i < n; i++)
        {
            float raw = target[i] + b[i]; // linear policy on [q - target, qd]; W = b = 0 for A
            for (int j = 0; j < 2 * n; j++)
                raw += W[i * 2 * n + j] * (j < n ? s[j] - target[j] : s[j]);
            float prev = smoothed[i];
            float blended = (1f - ActionSmooth) * prev + ActionSmooth * raw;
            float delta = Mathf.Clamp(blended - prev, -MaxDelta, MaxDelta);
            smoothed[i] = Mathf.Clamp(prev + delta, jointRanges[i].x, jointRanges[i].y);
        }
        return smoothed;
    }

    static float Gaussian()
    {
        return Mathf.Sqrt(-2f * Mathf.Log(1f - Random.value)) * Mathf.Cos(2f * Mathf.PI * Random.value);
    }

    public void ActionFrom(float[] destination)
    {
        System.Array.Copy(target, destination, target.Length);
    }

    // Home, or home + uniform(-span, span) clipped to the joint range (same as train_controller.sample_goal).
    public void PlaceTarget(bool home)
    {
        for (int i = 0; i < target.Length; i++)
        {
            if (home)
            {
                target[i] = homeTargets[i];
                continue;
            }
            float span = (jointRanges[i].y - jointRanges[i].x) * 0.5f * randomScale;
            target[i] = Mathf.Clamp(homeTargets[i] + Random.Range(-span, span), jointRanges[i].x, jointRanges[i].y);
        }
        hold = holdSeconds;

        if (smoothActions)
        {
            if (!home)
                targetCount++;
            int n = target.Length;
            W = new float[n * 2 * n]; // zeros: raw = target (A)
            b = new float[n];
            if (!home && targetCount % 2 == 0)
            {
                for (int k = 0; k < W.Length; k++)
                    W[k] = explorationStd * Gaussian();
                for (int k = 0; k < n; k++)
                    b[k] = 0.5f * explorationStd * Gaussian();
            }
        }
    }
}
