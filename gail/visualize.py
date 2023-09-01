
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers, MovieWriter
import matplotlib.animation as animation
import time
from matplotlib.patches import Wedge
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from itertools import chain
import argparse

import datetime as dt
import os

def parse():
    parser = argparse.ArgumentParser(description='video file name')
    parser.add_argument('--epochs', default='?', type=str, help='num of epochs')
    parser.add_argument('--lr', default='?', type= str, help='learning rate')
    parser.add_argument('--p', default='True', type=str, help= 'probability')
    parser.add_argument('--test', type=int, default=-1, help='test index')
    args = parser.parse_args()
    return args

def visualize(args):

    goal0 = np.load('gail/trained_data/goal0.npy')
    goal1 = np.load('gail/trained_data/goal1.npy')
    goal2 = np.load('gail/trained_data/goal2.npy')
    goals = [goal0, goal1, goal2]
    
    predictions0 = np.load('gail/trained_data/res0.npy')
    predictions1 = np.load('gail/trained_data/res1.npy')
    predictions2 = np.load('gail/trained_data/res2.npy')
    predictions = [predictions0, predictions1, predictions2]
    
    tables = np.load('gail/static_obs_states.npy')
    humans = np.load('gail/dynamic_obs_states.npy')
    human_controls = np.load('gail/dynamic_obs_controls.npy')
    
    task = tables.shape[0] + args.test
    
    traj_len, pred_len, n_pred, _ = predictions[0].shape
    
    table_pos = tables[task, :, :]
    trajectory = humans[task, 0, :, :]
    
    
    path = '~/anaconda3/bin/ffmpeg'
    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    writer = writers['ffmpeg'](fps=10)
    
    fig, ax = plt.subplots()
    
    ax.set_xlim(-2.5, 2.0)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    real_trajs = []
    for _ in range(humans.shape[1]):
        # ongoing trajectories of dynamic obstacles
        line, = ax.plot([], [], linestyle='dashed', linewidth=2, color='tab:grey', zorder=1)
        real_trajs.append(line)
        
        
        
    # tables
    for i in range(tables.shape[1]):
        obstacle = plt.Circle(table_pos[i],
                              .3,
                              ec=(0., 0., 0., 0.5),
                              fc=(0., 0., 1., 0.5),
                              lw=2,
                              )
        ax.add_artist(obstacle)
    
    #goals
    for g in goals:
         goal = plt.Circle(g, .1)
         ax.add_artist(goal)
    
    
    timestep = ax.text(2.5, -3.2, '', fontsize=15)
    
    pred_trajs0 = []
    for _ in range(n_pred):
        line, = ax.plot([], [], linestyle='solid', linewidth=2, color='tab:red', alpha=0.2, zorder=1)
        pred_trajs0.append(line)
        
    pred_trajs1 = []
    for _ in range(n_pred):
        line, = ax.plot([], [], linestyle='solid', linewidth=2, color='tab:red', alpha=0.2, zorder=1)
        pred_trajs1.append(line)
        
    pred_trajs2 = []
    for _ in range(n_pred):
        line, = ax.plot([], [], linestyle='solid', linewidth=2, color='tab:red', alpha=0.2, zorder=1)
        pred_trajs2.append(line)
    
    prev_people = []
        
    preds = [pred_trajs0, pred_trajs1, pred_trajs2]
        
    def func(t):
        
        for i, line in enumerate(real_trajs):
            line.set_data(humans[task, i, t:, 0], humans[task, i, t:, 1])
        
        for p_fig in list(prev_people):
                        p_fig.remove()
                        prev_people.remove(p_fig)
                    
        for i in range(humans.shape[1]):
            
            p_fig = ax.add_artist(plt.Circle(humans[task, i, t],
                                             .18, ec=(0., 0., 0., 0.5),
                                             fc=(1., 0., 0., 0.5),
                                             lw=2, zorder=2))
            prev_people.append(p_fig)
    
        
        for j, pred_t in enumerate(preds):
            for i, line in enumerate(pred_t):
                p = predictions[j]
                line.set_data(p[t, :, i, 0], p[t, :, i, 1])
            
            
        
        timestep.set_text('t = {:.2f}sec'.format(t * .1))
        
    fig.tight_layout()
    
    
    now = dt.datetime.now()
    video_dir = now.strftime("%Y-%m-%d")
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    file_name = f'{video_dir}/traj{tables.shape[0]}_e{args.epochs}_lr{args.lr}_test{args.test}is_scale{args.is_scale}.mp4'   
    ani = FuncAnimation(fig=fig, func=func, frames=traj_len)
    ani.save(file_name.format(time.strftime("%m%d-%H%M%S")), writer=writer, dpi=100)
    
if __name__ == "__main__":
    args = parse()
    
    
                                                