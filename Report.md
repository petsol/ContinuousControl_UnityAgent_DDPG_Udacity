## Introduction

This report details the implementation and hyperparameters used for the model used to solve the Reacher environment. The Reacher Environment was solved (reaching a mean score of 30. in the last 100 episodes) using DDPG algorithm in episode 508.

## Implementation of the DDPG algorithm

The DDPG algorithm uses an Actor-Critic, Local-Target network structure. The Actor is the main focus of the training, at inference, only the policies contained in the Actor(-Local) will be used. The Actor is in itself a continuous Proximal Policy Optimization algorithm, which takes the State as an input and transfers the data to the actual action-set that is a set (or a single) of continuous values (in this case the applicable joint-torque values). To aide the optimization process a Critic network is used, which is in itself a Deep Q Learning algorithm, trying to approximate the Q function for continuous state-action pairs. The difference between standalone DQN network and the Critic applied in DDPG is that the Critic only outputs a singular Q value, by taking a continuous action-set input. The two networks are paired together by virtue of relation between the Q values and the optimality of the actor policy. By maximalizing the Q value at state-action pairs the policy implemented can be optimized. For that reason, the gain function for the actor is the Q value for the present state obtained by the actor’s output for the present state (an action-set) and the present state itself. The loss function is the negative of the gain function. Using the loss function the gradient can be calculated through the critic network for the actor-network. Using these gradients (only) the actor-network will be optimized. The Critic network is optimized through the relatively independent actor-target network calculating the future action-set and through the sarsa max algorithm. The Critic network’s loss function is between the inferred (local) current state-action Q value and the expected Q value (for the present) obtained by the sarsa max algorithm from the future Q values, rewards, and gamma. To maintain relative-independence the expected Q value is calculated on the Critic-Target network. The loss function is the MSE of the two values, and it is backpropagated only on the Critic(-Local) network and used to optimize the Critic. After these operations, the target networks are soft updated with a tau coefficient at every Nth learning step. Both optimizers are ADAM, with fix learning rate. The DDPG algorithm also employs ExperienceBuffer, to smooth out gradients. To generate exploratory behavior a random noise is added to the Actor’s output during training, this Noise is a product of Ornstein-Uhlenbeck process, a highly correlative noise, and can be considered in itself as a random walk in continuous time.


## Implementation details

Batch normalization was not used in this project (code is commented out in the network sections). Through trial-and-error it was found that sampling every 3rd step only and storing in the memory speeds up the learning process substantially. The reason behind this might be the high correlation of states between successive training steps. The Actor-network has an input layer of state size (33), two hidden layers with 512 and 256 neurons respectively and an output layer of action-set size (4). The transfer functions except the output layer are all ReLU. The output layer has tangent-hyperbolic transfer function to correspond to the action intervals [-1, 1]. The Critic Network has similar structure except for an extra layer for action inputs, that is joined into the first hidden layer by direct addition to the input layers feed for each hidden layer neuron and ReLU-ed together before entering the first hidden layer (another possibility would have been to extend the first hidden layer with action size and join the action input there, but this showed better results). The output of the Critic network is a single neuron with a linear transfer function (no transfer function). The hidden layers are initialized in both networks with zero mean uniform distribution dependent on layer size.

The training parameters were low learning rates as 0.000075 for the actor and 0.0005 for the critic network. The soft update was applied at every learning step with a tau of 0.001. A batch size of 128 was used for training and for the critic network gamma was set at 0.9. The experience buffer size was 100,000.

The hyperparameter tuning was very hard, especially to find that a very low learning rate was necessary, and to realize that successive states are highly correlated meaning that subsampling the environment would give a boost to the learning process initially. Relatively low gamma was intuitive because of the continuous reward structure (when on track).


## Results

The above model with the mentioned hyperparameters, was able to converge to the desired target at episode 508, with a value of 30.0793.

![Continuous Control Convergence Graph](https://github.com/petsol/ContinuousControl_UnityAgent_DDPG_Udacity/blob/master/ContinuousControl_convergence.png?raw=true)

Udacity sources and Phil Tabor’s ‘LunarLanderContinuous-v2’ DDPG video was used as a model for the DDPG implementation.

## Future improvements
First by examining the low learning rate necessary to converge for the actor network, one can realize, that very smooth adjustment is needed for the joint torques to follow the target, however because of the high correlation between samples only every third step is taken to the Experience Buffer. The reason why this helped tremendously for training is possibly about the takeoff for the learning process. Overcorrelated samples hindered the initial learning and misguided the Critic network. To make use of all samples instead of only sampling every third sample, the learning should start much after the buffer has a single batch worth of data. If the buffer size to start learning would be raised to 10x the batch-size, initial sampling correlation could be minimized, and used episodes could be halved.
Another possible improvement would be to implement Prioritized Experience Replay. Now for the first glance this environment doesn't seem to be the good case to use PER, because of its continuous reward model, but there are two reasons it might help speed up the learning process. One is that initially random movements cross the target area not too often, the other is, that when it tracks the target area already a higher percentage of time, but looses traction eventually, the compensatory movements could be learned in a targeted fashion.
Other improvement would be to include a state-value Critic, because the state values here are relatively straightforward. This addition would possibly push the solution to top out at a much higher score than the original DDPG algorithm.

Sources:
- Ornstein–Uhlenbeck process:
   https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
- Udacity Deep Reinforcement Learning Git Repository:
   https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control
- Reinforcement Learning in Continuous Action Spaces | DDPG Tutorial (Pytorch) (Phil Tabor)
   https://www.youtube.com/watch?v=6Yd5WnYls_Y&t=2208s
