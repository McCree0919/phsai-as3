# Step 5 — Reflection

## 1. What were the hardest part for you, and what part was the easiest? How did you overcome this difficulty?

The hardest part was making IK reliable near the joint limits. Most targets solved cleanly, so at first I assumed the solver was fine. Then a low target exposed the problem: instead of failing loudly, the solver quietly settled about 118 mm away from the goal and reported that it was done. That told me it wasn't diverging; it was converging to the wrong answer, stuck against a joint limit it couldn't escape from its starting pose.

Since the starting pose was the trap, I gave the solver a second chance from a different one. When the first attempt fails to converge, it now solves again from the home pose. I keep the second result only if it lowers the predicted end-effector error by more than 1 mm, so the solver doesn't swap answers over noise.

The easiest part was setting up the target and markers, which only needed simple GameObjects and scene references. The bigger lesson came from the hard part: the easy poses will always look fine, so the real test of an IK solver is the poses near its limits.

## 2. How does your World Model work? What data is collected, what is your model architecture, what does your model predict?

I collected data in MuJoCo with two kinds of actions: 70,000 transitions with step targets and 50,000 with smooth actions. Step targets push the arm through sharp, fast changes, while smooth actions look more like what a controller actually sends, and I wanted the model to see both. Each row records six joint angles, six joint velocities, six actuator targets, and the resulting next angles and velocities.

The model is an MLP with 18 inputs, two hidden layers of 256 units with SiLU, and 12 outputs. Rather than predicting the next state directly, it predicts the change in joint state over one physics step; I undo the normalization and add that change back to the current state.

The design choice I thought about most was the loss. A model that is good at one step can still be bad at twenty, and imagined control rolls the model forward many times in a row. So I trained with a 20-step rollout loss alongside the one-step loss, to make the model pay for its own accumulated mistakes.

## 3. How does your controller work? How is it trained, how does it connect to your world model, what does it control?

The controller looks at how far each joint is from its goal and how fast it is moving, and uses that to correct the targets sent to the position actuators. It has 12 gains to tune.

Instead of tuning those gains on the real simulator, I let the world model do the imagining: CMA-ES proposes sets of gains, the world model predicts how the arm would respond, and the search keeps the gains that track best. L4.2 then smooths the commands and limits how much each one can change in a single physics step, so the arm never gets a sudden jolt.

The real check was moving back to reality. I loaded the selected gains into Unity and ran them with MuJoCo physics. On reachable joint goals, the controller cut the early tracking error compared with sending the goal directly, and by two seconds both approaches ended up close. So the benefit is mostly in how quickly the arm gets there, not where it ends up.

## 4. What does your IK solver do? How does it work and how does it connect to your controller? Why and how did you pick this algorithm?

The IK solver turns the red target sphere's position into six joint-angle goals for the Lite 6. I chose damped least squares because a plain Jacobian inverse can produce huge joint updates near a singularity, and the damping term keeps those updates bounded.

One detail mattered more than I expected: the solver uses MuJoCo's site Jacobian in a separate physics-data object. That way it can try out candidate poses without touching the live arm, which is what makes the retry-from-home strategy from Question 1 safe. If the first solve misses the tolerance, it tries again from home.

The solver's output becomes the input to the learned controller. IK decides where each joint should go, and the controller decides how to get there.

## 5. How has your understanding of World Models changed while doing this assignment?

Before this assignment, I thought of a world model mainly as a network that predicts the next state, and I assumed a small one-step error meant a good model. Training one changed that. On the step-target validation data, the epoch-7 model's joint-position error was only 0.000962 rad after one step, but it grew to 0.221034 rad after 100 steps, about 230 times larger. On the smooth-action data it grew even faster. Every prediction becomes the next input, so small errors compound.

What surprised me was that this imperfect model was still useful. It was good enough to train a controller, and the controller worked when I tested it in MuJoCo. That changed how I evaluate world models: prediction accuracy and usefulness for control are two different tests, and a model can fail the first while still passing the second.

## 6. How might you implement what you have learned here into your Term Project?

Our term project asks whether a climbing agent trained on generated structures can handle structures built by players. Our current agent learns by trial and error in the simulator. A world model could let it imagine instead: before committing to a reach or a weight shift, a controller could try several short action sequences in the model and pick the most promising one.

This assignment also tells me where that would get hard. The arm's joint motion is smooth and predictable, yet the error still compounded quickly. Climbing adds sudden events: a hand catching or slipping off an edge, or a box shifting under the climber's weight. I expect those contact changes, and unfamiliar structures, to be much harder to predict than joint motion. So I would start small: a short-horizon model for one climbing move, and a direct comparison between its predictions and what the simulator actually does, before trusting it to plan.
