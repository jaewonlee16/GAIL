import numpy as np
import gym
import torch
import gym_pedestrian
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.types import Trajectory
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env

import argparse
import gail.scaler as scaler

def parse():
    parser = argparse.ArgumentParser(description= 'enter id')
    parser.add_argument('--id', type = int, default = 0, help= 'enter 0 to 2')
    parser.add_argument('--epochs', type=int, default=300000, help= 'gail train epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate of GAIL')
    parser.add_argument('--test', type=int, default=-1, help='index of test evaluation trajectory')
    parser.add_argument('--s', type=bool, default=True, help='if scale from 0 to 1')
    args = parser.parse_args()
    return args



def generate_predictions(args, n_predictions: int, obs: np.ndarray, goal_prediction: np.ndarray, learner, n_steps, u_min, u_max, goals_prob, dt = 0.1):
    # return shape: (trajectory length, # predictions, 2)
    predictions = []

    # bayesian
    goals_pos = np.array([[-1.15, -0.25], [0.05, -0.25], [1.25, -0.25]])
    pr_pi = np.zeros(3)
    action = [0, 0, 0]


    for i, goal_pos in enumerate(goals_pos):

        o = torch.tensor(np.tile(np.concatenate([obs, goal_pos]), (n_predictions, 1)), dtype=torch.float32).to('cuda:0')
        action_torch, value, log_prob = learner.policy(o)
        #action[i] = np.array(action_torch.detach().cpu())
        log_prob = np.array(log_prob.detach().cpu())
        pr_pi[i] = np.exp(log_prob.mean())
        

    numerator = pr_pi * goals_prob
    denominator = np.sum(numerator)
    goals_prob = numerator / denominator

    #a = action[np.argmax(goals_prob)]

    goal_prediction = goals_pos[np.argmax(goals_prob)]
    o = torch.tensor(np.tile(np.concatenate([obs, goal_prediction]), (n_predictions, 1)), dtype=torch.float32).to('cuda:0')
    for _ in range(n_steps):
        predictions.append(o[:, :2].detach().cpu().numpy())
        scaled_o = scaler.scale_obs(o.detach().cpu(), is_scale=args.is_scale, isZero2One=args.s).to('cuda:0')

        scaled_a, _, _ = learner.policy(scaled_o)
        a = scaler.inverse_u(scaled_a, args.is_scale,u_min, u_max)
        # print(a.shape)
        x, y, th, lin_v, ang_v, x_goal, y_goal = o[:, 0], o[:, 1], o[:, 2], o[:, 3], o[:, 4], o[:, 5], o[:, 6]
        #x, y, th, lin_v, ang_v, vpref, x_goal, y_goal = o[:, 0], o[:, 1], o[:, 2], o[:, 3], o[:, 4], o[:, 5], o[:, 6], o[:, 7]
        lin_a, ang_a = a[:, 0], a[:, 1]
        x_next = x + dt * lin_v * torch.cos(th)
        y_next = y + dt * lin_v * torch.sin(th)
        th_next = th + dt * ang_v
        th_next = (th_next + np.pi) % (2 * np.pi) - np.pi #th_next normalization
                                              #theta must be in range [-pi, pi]
        lin_v_next = torch.clamp(lin_v + dt * lin_a, 0, .16)
        ang_v_next = torch.clamp(ang_v + dt * ang_a, -.5, .5)
        o = torch.stack([x_next, y_next, th_next, lin_v_next, ang_v_next, x_goal, y_goal], dim=-1)
        

    return [np.array(predictions), goals_prob, goal_prediction]




def gail_train(args, id = 0):

    rng = np.random.default_rng(0)

    env = gym.make("Pedestrian-v0")
    """
    expert = PPO(policy=MlpPolicy, env=env, n_steps=64)
    expert.learn(1000)

    rollouts = rollout.rollout(
        expert,
        make_vec_env(
            "Pedestrian-v0",
            n_envs=5,
            post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
        rng=rng,
        ),
        rollout.make_sample_until(min_timesteps=None, min_episodes=60),
        rng=rng,
    )
    """
    rollouts = []
    x = np.load('gail/dynamic_obs_states.npy')

    # u = np.load('dynamic_obs_controls.npy')
    
    u = (x[:, :, 1:, 3:] - x[:, :, :-1, 3:]) / env.dt 
    #print(env.dt)
    n_tasks, n_obs, n_steps, n_dim = x.shape
    x_goal = np.tile(x[:, :, -1:, :2], (1, 1, n_steps, 1))
    x = np.concatenate([x, x_goal], axis=-1) # x,y, theta, linv, angv, vpref, goalx, goaly
    #print(n_tasks)
    n_train_tasks = round(n_tasks * 0.9)
    n_eval_tasks = n_tasks - n_train_tasks

    #id = args.id
    #print(x.shape)  #(? ex3000, 3, 201, 7)

    original_x = x
    x = scaler.scale_obs(x, is_scale=args.is_scale, isZero2One=args.s)
    u_min = np.min(u)
    u_max = np.max(u)
    scaled_u = scaler.scale_u(u, args.is_scale,u_min, u_max)
    for id in range(n_obs):    
        for i in range(n_train_tasks):
            obs = x[i, id, :, :]
            acts = scaled_u[i, id, :, :]
            t = Trajectory(obs, acts, None, False)
            #print(f"{t=}")
            rollouts.append(t)


    venv = make_vec_env("Pedestrian-v0", n_envs=8, rng=rng)
    learner = PPO(env=venv, policy=MlpPolicy, learning_rate=args.lr)

    reward_net = BasicRewardNet(
        venv.observation_space,
        venv.action_space,
        normalize_input_layer=RunningNorm,
    )
    gail_trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
    )

    gail_trainer.train(args.epochs)
    rewards, _ = evaluate_policy(learner, venv, 100, return_episode_rewards=True)
    # learner.learn(10000)
    # print("Rewards:", rewards)


    for id in range(n_obs):
        x = original_x
        # shape: (trajectory length + 1, 5)
        obs_eval = x[args.test, id, :, :5]

        goal_pred = np.mean(x[:n_train_tasks, id, -1, :2], axis=0)
        np.save(f'{args.result_path}goal{id}.npy', goal_pred)
        
        res = []
        """
        # adding vpref to goal_pred at test time
        vpref_estimate = np.mean(x[args.test, id, 5:15, 3]) # estimate timestep from 5 to 15
        goal_pred = np.concatenate([vpref_estimate, goal_pred], axis=None)
        """

        goal_result = []
        goals_prob = np.array([0.5, 0.2, 0.3]) # used for bayesian
        for t in range(n_steps):
            obs_t = obs_eval[t]
            p_t , goals_prob, goal_prediction = generate_predictions(n_predictions=10, 
                                       obs=obs_t, 
                                       goal_prediction=goal_pred, 
                                       learner=learner, 
                                       n_steps=n_steps//1, 
                                       dt = env.dt,
                                       args = args,
                                        u_min=u_min,
                                        u_max=u_max,
                                        goals_prob=goals_prob
                                    )
            res.append(p_t)
            goal_result.append(goal_prediction)

        # shape: (trajectory length + 1, trajectory length, # predictions, 2)
        np.save(f'{args.result_path}res{id}.npy', res)
        np.save('result_goal.npy', goal_result)
        
    
if __name__ == "__main__":

    args = parse()