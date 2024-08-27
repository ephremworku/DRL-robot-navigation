import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
# from IPython.display import clear_output
import matplotlib.pyplot as plt

from velodyne_env import GazeboEnv
# %matplotlib inline
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
# from common.multiprocessing_env import SubprocVecEnv

environment_dim = 20
robot_dim = 4
max_action = 1
random_near_obstacle=True

# envs = [make_env() for i in range(num_envs)]
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)

# env = gym.make(env_name)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)
        

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value
        
def plot(frame_idx, rewards):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()
    
def test_env(vis=False):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        action = dist.sample().clamp(-max_action, max_action).cpu().numpy()[0]
        if random_near_obstacle:
            if (
                np.random.uniform(0, 1) > 0.85
                and min(state[4:-8]) < 0.6
                and count_rand_actions < 1
            ):
                count_rand_actions = np.random.randint(8, 15)
                random_action = np.random.uniform(-1, 1, 2)

            if count_rand_actions > 0:
                count_rand_actions -= 1
                action = random_action
                action[0] = -1
        a_in = [(action[0] + 1) / 2, action[1]]

        next_state, reward, done, _ = env.step(a_in)
        state = next_state
        total_reward += reward
    return total_reward
    
def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns
    
def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        
        

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
num_inputs  = 24
num_outputs = 2

#Hyper params:
hidden_size      = 256
lr               = 3e-4
num_steps        = 20
mini_batch_size  = 5
ppo_epochs       = 4
threshold_reward = -200

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

max_frames = 15000
frame_idx  = 0
test_rewards = []

state = env.reset()
early_stop = False

while frame_idx < max_frames and not early_stop:

    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    masks     = []
    entropy = 0

    for _ in range(num_steps):
        state = torch.FloatTensor(state).to(device)
        dist, value = model(state)
        action = dist.sample().clamp(-max_action, max_action).cpu().numpy()
        if random_near_obstacle:
            if (
                np.random.uniform(0, 1) > 0.85
                and min(state[4:-8]) < 0.6
                and count_rand_actions < 1
            ):
                count_rand_actions = np.random.randint(8, 15)
                random_action = np.random.uniform(-1, 1, 2)

            if count_rand_actions > 0:
                count_rand_actions -= 1
                action = random_action
                action[0] = -1
        a_in = [(action[0] + 1) / 2, action[1]]

        # action = dist.sample()
        next_state, reward, done, _ = env.step(a_in)

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()
        
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
        
        states.append(state)
        actions.append(action)
        
        state = next_state
        frame_idx += 1
        
        if frame_idx % 1000 == 0:
            test_reward = np.mean([test_env() for _ in range(10)])
            test_rewards.append(test_reward)
            plot(frame_idx, test_rewards)
            if test_reward > threshold_reward: early_stop = True
            

    next_state = torch.FloatTensor(next_state).to(device)
    _, next_value = model(next_state)
    returns = compute_gae(next_value, rewards, masks, values)

    returns   = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values    = torch.cat(values).detach()
    states    = torch.cat(states)
    actions   = torch.cat(actions)
    advantage = returns - values
    
    ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)
    

from itertools import count

max_expert_num = 50000
num_steps = 0
expert_traj = []

for i_episode in count():
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        action = dist.sample().clamp(-max_action, max_action)
        action = dist.sample().cpu().numpy()[0]
        if random_near_obstacle:
            if (
                np.random.uniform(0, 1) > 0.85
                and min(state[4:-8]) < 0.6
                and count_rand_actions < 1
            ):
                count_rand_actions = np.random.randint(8, 15)
                random_action = np.random.uniform(-1, 1, 2)

            if count_rand_actions > 0:
                count_rand_actions -= 1
                action = random_action
                action[0] = -1
        a_in = [(action[0] + 1) / 2, action[1]]
        
        next_state, reward, done, _ = env.step(a_in)
        state = next_state
        total_reward += reward
        expert_traj.append(np.hstack([state, action]))
        num_steps += 1
    
    print("episode:", i_episode, "reward:", total_reward)
    
    if num_steps >= max_expert_num:
        break
        
expert_traj = np.stack(expert_traj)
print()
print(expert_traj.shape)
print()
np.save("expert_traj.npy", expert_traj)
