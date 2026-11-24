# Step 5 — Reflection

## 1. What were the hardest part for you, and what part was the easiest? How did you overcome this difficulty?

The hardest part was getting IK to work reliably near joint limits. In an early test, a low target left the solver stuck about 118 mm away. I added a second solve from the home pose when the first attempt failed to converge, and kept the second result when it reduced the predicted end-effector error by more than 1 mm. Setting up the target and markers was easier because they only needed simple GameObjects and scene references. The low target taught me to test poses near the limits as well as poses that are easy to reach.

## 2. How does your World Model work? What data is collected, what is your model architecture, what does your model predict?

I collected 70,000 transitions with step targets and 50,000 with smooth actions in MuJoCo. Each row contains six joint angles, six joint velocities, six actuator targets, and the next angles and velocities. My world model is an MLP with 18 inputs, two hidden layers of 256 units using SiLU, and 12 outputs. It predicts the change in joint state over one physics step, which I add back to the current state after undoing the normalization. I used a 20-step loss alongside the one-step loss because imagined control rolls the model forward repeatedly.

## 3. How does your controller work? How is it trained, how does it connect to your world model, what does it control?

My controller uses the difference between the current joint angles and the goal, along with joint velocities, to correct the targets sent to the position actuators. I used CMA-ES to tune 12 gains while the world model simulated the results of the controller's actions. L4.2 smooths the commands and limits how much each one can change in a physics step. I loaded the selected gains into Unity and tested the controller with MuJoCo physics. On reachable joint goals, it reduced the early tracking error compared with direct goal control, and the errors were close at two seconds.

## 4. What does your IK solver do? How does it work and how does it connect to your controller? Why and how did you pick this algorithm?

My IK solver converts the red target sphere's position into six joint-angle goals for the Lite 6. I chose damped least squares because damping keeps the joint updates from getting too large near a singularity. It uses MuJoCo's site Jacobian in a separate physics-data object so it can test poses independently of the live arm. If the first solve misses the tolerance, the solver tries again from home. It passes the resulting joint goals to the learned controller, which sends commands to the actuators.

## 5. How has your understanding of World Models changed while doing this assignment?

Before this assignment, I mainly thought of a world model as a network that predicts the next state. Training one showed me how much error can build up when each prediction becomes the input for the next. On the step-target validation data, the epoch-7 model's joint-position error grew from 0.000962 rad after one step to 0.221034 rad after 100 steps, and it grew faster on the smooth-action data. I could still use the model to train a controller, then check the controller's behavior in MuJoCo. I now see those as two different tests, one for prediction and one for control.

## 6. How might you implement what you have learned here into your Term Project?

Our term project studies whether a climbing agent trained on generated structures can handle structures built by players. I could use a world model to predict what happens when the climber reaches for a handhold or shifts its body. A controller could try several short action sequences in the model before choosing one in the simulator. I expect changes in hand contact and unfamiliar structures to be harder to predict than the arm's joint motion. I would start with a small climbing prototype and compare the predicted motions with what the simulator actually does.
