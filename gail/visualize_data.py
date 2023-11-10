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
import argparse

def visualize_data(args, task = -1):
    tables = np.load('gail/static_obs_states.npy')
    humans = np.load('gail/dynamic_obs_states.npy')



    # parameters
    traj_len = 200
    table_pos = tables[-1, :, :]

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
                              args.table_radii/2,
                              ec=(0., 0., 0., 0.5),
                              fc=(0., 0., 1., 0.5),
                              lw=2,
                              )
        ax.add_artist(obstacle)
    

    
    
    timestep = ax.text(2.5, -3.2, '', fontsize=15)


    prev_people = []

        
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
    

            
            
        
        timestep.set_text('t = {:.2f}sec'.format(t * .1))
        
    fig.tight_layout()
    
    
    now = dt.datetime.now()
    video_dir = now.strftime("%Y-%m-%d")
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    file_name = f'{video_dir}/fast_traj{tables.shape[0]}_task{task}_lr{args.lr}_test{args.test}.mp4'   
    ani = FuncAnimation(fig=fig, func=func, frames=traj_len)
    ani.save(file_name.format(time.strftime("%m%d-%H%M%S")), writer=writer, dpi=100)
    

def parse():
    parser = argparse.ArgumentParser(description= 'enter options')
    parser.add_argument('--table_radii', required=False, default=0.5, type=float)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()  
    for task in [-3, -4, -5]:
        visualize_data(args, task)
