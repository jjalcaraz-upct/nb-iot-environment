# dqn agent based on clean-rl implementation
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from gymnasium import spaces

device = "cpu"

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, obs_shape, action_n):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(obs_shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_n),
        )

    def forward(self, x):
        return self.network(x)
    

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class DQN_Agent():
    def __init__(self,
                 obs_shape, 
                 action_n,
                 total_timesteps: int = 500000,
                 seed: int = 123,
                 torch_deterministic: bool = True,
                 learning_rate: float = 2.5e-4,
                 num_envs: int = 1,
                 buffer_size: int = 10000,
                 gamma: float = 0.99,
                 tau: float = 1.0,
                 target_network_frequency: int = 500,
                 batch_size: int = 128,
                 start_e: float = 1,
                 end_e: float = 0.05,
                 exploration_fraction: float = 0.5,
                 learning_starts: int = 10000,
                 train_frequency: int = 10):
        """
        Initialize the algorithm with specified parameters.

        :param total_timesteps: total timesteps of the experiments
        :param learning_rate: the learning rate of the optimizer
        :param num_envs: the number of parallel game environments
        :param buffer_size: the replay memory buffer size
        :param gamma: the discount factor gamma
        :param tau: the target network update rate
        :param target_network_frequency: the timesteps it takes to update the target network
        :param batch_size: the batch size of sample from the reply memory
        :param start_e: the starting epsilon for exploration
        :param end_e: the ending epsilon for exploration
        :param exploration_fraction: the fraction of `total-timesteps` it takes from start-e to go end-e
        :param learning_starts: timestep to start learning
        :param train_frequency: the frequency of training
        """
        self.seed = seed
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau
        self.target_network_frequency = target_network_frequency
        self.batch_size = batch_size
        self.start_e = start_e
        self.end_e = end_e
        self.exploration_fraction = exploration_fraction
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency
        self.single_action_space = spaces.Discrete(action_n)
        self.auxiliary_action_space = spaces.Discrete(action_n)
        self.num_actions = action_n # actions without the lagrange multipler
  
        # REVISAR ESTA PARTE 
        # PARAR UN GENERADOR RNG COMO ARG
        # seeding
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = torch_deterministic

        #  q_network
        self.q_network = QNetwork(obs_shape, action_n).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr = self.learning_rate, eps=1e-5)

        # target_network
        self.target_network = QNetwork(obs_shape, action_n).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # replay buffer
        self.rb = ReplayBuffer(
            self.buffer_size,
            spaces.Box(0, 1, shape = obs_shape), # create an observation space
            self.single_action_space,
            device,
            handle_timeout_termination=False,
        )

        # execution control
        self.step_counter = 0
        self.iteration = 0
        self.next_action = None
        self.next_logprobs = None
        self.next_values = None
        self.next_obs = None


    def get_first_action(self):
        return self.auxiliary_action_space.sample()

    def store_transition(self, obs, next_obs, action, r):
        # obs = np.array(obs, dtype=float)
        self.rb.add(obs, next_obs, action, r, False, {})

    def get_action(self, obs, index = 0):
        # obs = np.array(obs, dtype=float)
        step = self.step_counter
        
        epsilon = linear_schedule(self.start_e, self.end_e, self.exploration_fraction * self.total_timesteps, step)
        if random.random() < epsilon:
            action = self.auxiliary_action_space.sample()
        else:
            # print(f'obs: {obs}')
            q_values = self.q_network(torch.Tensor(obs).to(device))
            start_ = index * self.num_actions
            end_ = start_ + self.num_actions
            q_values = q_values[start_:end_]
            action = torch.argmax(q_values, dim=0).cpu().numpy()

        # ALGO LOGIC: training
        if step > self.learning_starts:
            if step % self.train_frequency == 0:
                self.optimize_network()
            if step % self.target_network_frequency == 0:
                self.update_target()

        self.step_counter += 1

        return action
    

    def optimize_network(self):
        data = self.rb.sample(self.batch_size)
        with torch.no_grad():
            target_max, _ = self.target_network(data.next_observations).max(dim=1)
            td_target = data.rewards.flatten() + self.gamma * target_max * (1 - data.dones.flatten())
        old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
        loss = F.mse_loss(td_target, old_val)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_target(self):
        for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_network_param.data.copy_(
                self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data
            )

