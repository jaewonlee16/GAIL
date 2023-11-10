# export PYTHONPATH="/home/core/forces_pro_client":$PYTHONPATH
import argparse
from gail.example import gail_train
from data_generation.indoor import Indoor
from gail.visualize import visualize
from gail.visualize_data import visualize_data

import shutil
import glob
import os


def parse():
    parser = argparse.ArgumentParser(description= 'enter options')

    parser.add_argument('--generate_data', required=False, action='store_true', help='if generate data')
    # data_generation
    parser.add_argument('--num_tasks', required=False, default=10, type=int)
    parser.add_argument('--num_tables', required=False, default=1, type=int)
    parser.add_argument('--num_people', required=False, default=0, type=int)
    parser.add_argument('--table_radii', required=False, default=0.5, type=float)
    parser.add_argument('--robot_rad', required=False, default=0.455, type=float)
    parser.add_argument('--human_radii', required=False, default=0.1, type=float)
    parser.add_argument('--ep_len', required=False, default=200, type=int)
    parser.add_argument('--dt', required=False, default=0.1, type=float)
    parser.add_argument('--print_map', action='store_true')

    # gail
    #parser.add_argument('--id', type = int, default = 0, help= 'enter 0 to 2')
    parser.add_argument('--epochs', type=int, default=300000, help= 'gail train epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate of GAIL')
    parser.add_argument('--test', type=int, default=-1, help='index of test evaluation trajectory')
    parser.add_argument('--is_scale', action='store_true', help='if train with scaled data or not')
    parser.add_argument('--s', type=bool, default=True, help='if scale from 0 to 1')
    parser.add_argument('--result_path', default= 'gail/trained_data/')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()
    # data generation
    if args.generate_data:
        map = Indoor(
            num_tables=args.num_tables,
            num_tasks=args.num_tasks,
            num_people=args.num_people,
            table_radii=args.table_radii,
            human_radii=args.human_radii,
            robot_rad=args.robot_rad,
            ep_len=args.ep_len,
            dt=args.dt
        )
        
        for file in glob.glob('data_generation/data/*.npy'):
            shutil.move(file, 'gail')

        for task in [-1, -2, -3]:
            visualize_data(args, task)

    else:
        # gail train
        if not os.path.exists(args.result_path):
            os.makedirs(args.result_path)
        
        gail_train(args)

        # visualize
        visualize(args)