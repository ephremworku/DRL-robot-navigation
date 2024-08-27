import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from velodyne_env import GazeboEnv


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        # Function to load network parameters
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )


# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
seed = 0  # Random seed number
max_ep = 500  # maximum number of steps per episode
file_name = "td3_velodyne"  # name of the file to load the policy from


# Create the testing environment
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2



way_points = [(u[0], u[1]) for u in np.loadtxt('output.txt')]


way_points_or_not = False
robot_initial_heading_angle = -np.loadtxt('heading.txt')

# Create the network
network = TD3(state_dim, action_dim)
try:
    network.load(file_name, "./pytorch_models")
except:
    raise ValueError("Could not load the stored model parameters")

done = True
episode_timesteps = 0
# at the begining of the robot this argument passes 3 parameters:
# Robot initial point: way_points[0]
# initial way point: way_points[1]
# Robot initial orientation: robot_intial_heading_angle
if not way_points_or_not:
    state = env.reset(way_points[0], way_points[-1])
else:
    state = env.reset(way_points[0], way_points[1], robot_initial_heading_angle, False)

way_points_counter = 1

# Begin the testing loop
while True:
    if way_points_or_not:
        action = network.get_action(np.array(state))
        way_points_to_pass = way_points[way_points_counter]

        # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
        a_in = [(action[0] + 1) / 3, action[1]]
        next_state, reward, done, target, collision = env.step(a_in, way_points_to_pass)
        done = 1 if episode_timesteps + 1 == max_ep else int(done)
        if target:
            way_points_counter+=1
            if way_points_counter == len(way_points):
                state = env.reset()
                done = False
                episode_timesteps = 0
                way_points_counter = 0


        # On termination of episode
        if collision:
            state = env.reset()
            done = False
            episode_timesteps = 0
        else:
            state = next_state
            episode_timesteps += 1
    else:
        action = network.get_action(np.array(state))
        way_points_to_pass = way_points[way_points_counter]

        # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
        a_in = [(action[0] + 1) / 2, action[1]]
        next_state, reward, done, target, collision = env.step(a_in)
        done = 1 if episode_timesteps + 1 == max_ep else int(done)
        if target:
            way_points_counter+=1
            if way_points_counter == len(way_points):
                state = env.reset()
                done = False
                episode_timesteps = 0
                way_points_counter = 0


        # On termination of episode
        if collision:
            state = env.reset()
            done = False
            episode_timesteps = 0
        else:
            state = next_state
            episode_timesteps += 1

