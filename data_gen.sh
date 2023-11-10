
export PYTHONPATH="/home/core/forces_pro_client":$PYTHONPATH

#cp gail/dynamic_obs_states.npy gail/static_obs_states.npy gail/dynamic_obs_controls.npy gail/second_data/
rm gail/dynamic_obs_states.npy gail/static_obs_states.npy gail/dynamic_obs_controls.npy
python3 train.py --generate_data --num_tasks 3000
